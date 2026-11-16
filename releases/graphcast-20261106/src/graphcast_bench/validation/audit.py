"""Keep the primary training failure when dispatch auditing also fails."""

import json
import sys


def run_with_audit(train, save_audit, path, on_failure=None):
    try:
        result = train()
    except BaseException as primary:
        if on_failure is not None:
            try:
                on_failure(primary)
            except Exception as diagnostic_error:
                print(
                    "Unable to save failure diagnostic: " + repr(diagnostic_error), file=sys.stderr
                )
        try:
            save_audit(path)
        except Exception as audit_error:
            diagnostic = dict(
                status="incomplete_after_training_error",
                primary_error=repr(primary),
                audit_error=repr(audit_error),
            )
            try:
                path.write_text(json.dumps(diagnostic, indent=2) + "\n")
            except Exception as write_error:
                print("Unable to save incomplete audit: " + repr(write_error), file=sys.stderr)
        raise
    save_audit(path)
    return result
