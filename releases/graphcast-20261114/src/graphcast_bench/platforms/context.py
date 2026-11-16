"""Explicit per-process application settings; host Python needs only stdlib."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

SCHEMA = 'graphcast.runtime_context.v1'


class RuntimeContext:
    def __init__(self, kind, platform, output, device, paths, config, debug=None):
        if kind not in ('training', 'inference') or platform not in ('bw1100', 'h100'):
            raise ValueError('Unknown runtime kind/platform')
        if not isinstance(output, (str, Path)) or not Path(output).is_absolute():
            raise ValueError('Runtime output must be an absolute path')
        if (not isinstance(device, dict) or type(device.get('index')) is not int
                or device['index'] < 0 or not isinstance(device.get('pci'), str)
                or not re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', device['pci'])):
            raise ValueError('Runtime requires an explicit physical device and PCI address')
        if not isinstance(paths, dict) or not isinstance(config, dict):
            raise ValueError('Runtime paths/config must be objects')
        required = ['assets'] + (['training_data'] if kind == 'training' else [])
        if platform == 'bw1100':
            required += [kind + '_graph', kind + '_router']
        for name in required:
            if not isinstance(paths.get(name), str) or not Path(paths[name]).is_absolute():
                raise ValueError('Missing absolute runtime path: ' + name)
        libraries = paths.get('library_dirs', [])
        if (not isinstance(libraries, list) or any(
                not isinstance(path, str) or not Path(path).is_absolute() for path in libraries)):
            raise ValueError('Runtime library_dirs must be a list of absolute paths')
        settings = dict(trace=kind == 'training' and platform == 'bw1100', compile_only=False,
                        fp32_reference=None)
        if debug is not None and not isinstance(debug, dict):
            raise ValueError('Runtime debug controls must be an object')
        settings.update(debug or {})
        if any(type(settings[name]) is not bool for name in ('trace', 'compile_only')):
            raise ValueError('Runtime trace/compile_only must be booleans')
        reference = settings['fp32_reference']
        if reference is not None and (not isinstance(reference, str) or not Path(reference).is_absolute()):
            raise ValueError('FP32 reference must be an absolute path')
        self.kind, self.platform, self.output = kind, platform, Path(output)
        self.device, self.paths, self.config = deepcopy(device), deepcopy(paths), deepcopy(config)
        self.device['pci'] = self.device['pci'].lower()
        self.debug = settings

    def path(self, name):
        return Path(self.paths[name])

    def require_kind(self, kind):
        if self.kind != kind:
            raise ValueError('Runtime context kind does not match worker: ' + kind)
        return self

    def to_dict(self):
        return deepcopy(dict(schema=SCHEMA, kind=self.kind, platform=self.platform,
                             output=str(self.output), device=self.device, paths=self.paths,
                             config=self.config, debug=self.debug))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('schema') != SCHEMA:
            raise ValueError('Invalid runtime context schema')
        try:
            return cls(**{name: value[name] for name in
                          ('kind', 'platform', 'output', 'device', 'paths', 'config', 'debug')})
        except KeyError as error:
            raise ValueError('Missing runtime context field: ' + str(error))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def write(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n')
        temporary.replace(path)


def context_arguments(argv=None):
    """Read context before accelerator imports; leave worker-specific flags intact."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runtime-context', type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    return RuntimeContext.load(args.runtime_context), remaining
