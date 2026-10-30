#!/usr/bin/env bash
set +e
"$@"
child_code=$?
printf '%s\n' "$child_code" > "$TRAINING_OUTPUT/python-exit-code.txt"
exit "$child_code"
