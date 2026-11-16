#!/usr/bin/env bash
# Run from a delivered source package. No personal paths or existing containers required.
set -euo pipefail

fail() { printf 'GraphCast: %s\n' "$*" >&2; exit 1; }
usage() {
  printf '%s\n' '用法: bash environment/container.sh DELIVERY_ROOT WORK_ROOT [-- COMMAND ...]' \
    '首次创建环境，此后复用当前账户的容器；不指定 COMMAND 时进入 Bash。' \
    'DELIVERY_ROOT 内含源码包、runtime/、data/ 和 SHA256SUMS。' \
    'WORK_ROOT 是当前账户自己的工作目录，用于配置、构建和结果。'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
delivery=$(cd -- "$1" 2>/dev/null && pwd -P) || fail '交付目录不存在或不可访问'
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $(dirname -- "$package") == "$delivery" ]] || fail '源码包必须是交付目录的直接子目录'
release=$(basename -- "$package")
[[ $release =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || fail '源码目录名仅支持字母、数字、点、下划线和短横线'
for entry in runtime/work1 runtime/public runtime/opt/flagos data; do
  [[ -d $delivery/$entry && -r $delivery/$entry && -x $delivery/$entry ]] || fail "缺少或无法读取 $entry"
done
[[ -r $delivery/SHA256SUMS && -r $package/release.json ]] || fail '缺少配套校验清单或源码 release.json'
[[ -d /opt/hyhal && -e /dev/kfd && -d /dev/dri ]] || fail '需要已安装匹配海光驱动的主机'
host_uid=$(id -u)
host_gid=$(id -g)
umask 077
mkdir -p -- "$2"
work=$(cd -- "$2" && pwd -P)
[[ $(stat -c %u -- "$work") == "$host_uid" && -w $work ]] || fail '工作目录必须归当前账户所有且可写'
[[ $work != / && $work != "$delivery" && $work != "$package" ]] || fail '请使用单独的工作目录'
[[ $work != "$delivery/"* && $delivery != "$work/"* ]] || fail '工作目录与交付目录不能互相包含'
for path in "$delivery" "$work"; do
  [[ $path != *','* && $path != *'|'* && $path != *$'\n'* ]] || fail '目录路径不能包含逗号、竖线或换行'
done
if [[ -e $work/.graphcast-work ]]; then
  [[ $(cat "$work/.graphcast-work") == "$host_uid:$host_gid" ]] || fail '工作目录账户标识不匹配'
else
  [[ -z $(find "$work" -mindepth 1 -maxdepth 1 -print -quit) ]] || fail '首次使用请选择空的专用工作目录'
  printf '%s:%s\n' "$host_uid" "$host_gid" >"$work/.graphcast-work"
fi
shift 2
if [[ $# -gt 0 ]]; then
  [[ $1 == -- ]] || fail '附加命令前请加 --'
  shift
fi
exec 9>"$work/.container.lock"
flock -n 9 || fail '此工作目录已有活动会话，请先退出该会话'
docker info >/dev/null 2>&1 || fail '当前账户无法连接 Docker；请先由管理员开通 Docker 使用权限'

runtime_image=sha256:6d4813ee61bb39a439d96bb28b62371628d9eb0a2e62cc4a84a5218403bb0aad
container="graphcast-${host_uid}-6d4813ee"
identity="v2|$host_uid|$host_gid|$delivery|$work"
if docker container inspect "$container" >/dev/null 2>&1; then
  actual=$(docker inspect --format '{{index .Config.Labels "graphcast.identity"}}' "$container")
  [[ $actual == "$identity" ]] || fail "容器 $container 的目录配置不同；请沿用原目录，或结束旧会话后手动移除该容器再创建"
  [[ $(docker inspect --format '{{.Image}}' "$container") == "$runtime_image" ]] || fail '已有容器镜像版本不符'
else
  (cd -- "$delivery" && sha256sum --quiet -c SHA256SUMS) || fail '配套材料校验失败'
  if ! docker image inspect "$runtime_image" >/dev/null 2>&1; then
    [[ -r $delivery/runtime-image.tar ]] || fail '固定镜像未安装，且缺少 runtime-image.tar'
    docker load -i "$delivery/runtime-image.tar"
  fi
  [[ $(docker image inspect --format '{{.Id}}' "$runtime_image") == "$runtime_image" ]] || fail '镜像版本不符'
  monitoring=()
  if [[ -e /dev/mkfd ]]; then monitoring+=(--device=/dev/mkfd); fi
  if [[ -S /run/hycu_sk ]]; then
    monitoring+=(--mount 'type=bind,src=/run/hycu_sk,dst=/run/hycu_sk,readonly')
  fi
  docker create --name "$container" --label "graphcast.identity=$identity" "${monitoring[@]}" \
    --user 0:0 --device=/dev/kfd --device=/dev/dri --ipc=host \
    --mount "type=bind,src=$delivery/runtime/work1,dst=/work1,readonly" \
    --mount "type=bind,src=$delivery/runtime/public,dst=/public,readonly" \
    --mount "type=bind,src=$delivery/runtime/opt/flagos,dst=/opt/flagos,readonly" \
    --mount 'type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly' \
    --mount "type=bind,src=$delivery,dst=/delivery,readonly" \
    --mount "type=bind,src=$delivery/data,dst=/data,readonly" \
    --mount "type=bind,src=$work,dst=/workspace" \
    --env PYTHONDONTWRITEBYTECODE=1 "$runtime_image" sleep infinity >/dev/null
fi
if [[ $(docker inspect --format '{{.State.Running}}' "$container") != true ]]; then
  docker start "$container" >/dev/null
fi

# GPU uses the validated root runtime; files on the host belong to the caller.
return_files() {
  local result=$?
  trap - EXIT
  if ! docker exec "$container" chown -R -h "$host_uid:$host_gid" /workspace; then
    printf 'GraphCast: 文件所有权恢复失败；恢复 Docker 后重新运行本入口。\n' >&2
    [[ $result -ne 0 ]] || result=1
  fi
  exit "$result"
}
trap return_files EXIT
docker exec "$container" chown -R -h "$host_uid:$host_gid" /workspace
terminal=(-i)
if [[ -t 0 && -t 1 ]]; then terminal+=(-t); fi
printf '容器: %s\n工作目录: %s\n' "$container" "$work"
docker exec "${terminal[@]}" --workdir "/delivery/$release" "$container" \
  /bin/bash environment/container-session.sh "$@"
