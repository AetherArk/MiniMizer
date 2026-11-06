"""Versioned JSON contracts; molecular validation remains in the workflow."""
import json
import math
from pathlib import Path
from jsonschema import Draft202012Validator, validators

# Index consumers require integer JSON tokens, not integral floats or booleans.
Validator = validators.extend(Draft202012Validator, type_checker=
    Draft202012Validator.TYPE_CHECKER.redefine('integer', lambda _, value: type(value) is int))

SCHEMA_VERSION = '1.0'
SCHEMAS = Path(__file__).resolve().parents[1] / 'schemas'


def validate(value, kind):
    def finite(item, path):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f'{kind}{path}: nonfinite number')
        if isinstance(item, dict):
            for key, child in item.items():
                finite(child, f'{path}.{key}')
        elif isinstance(item, list):
            for index, child in enumerate(item):
                finite(child, f'{path}[{index}]')
    finite(value, '')
    schema = json.loads((SCHEMAS / f'{kind}.schema.json').read_text())
    error = next(Validator(schema).iter_errors(value), None)
    if error:
        path = '.'.join(map(str, error.absolute_path))
        # Do not echo an entire malformed scientific packet.
        raise ValueError(f'{kind}.{path or "$"}: {error.validator} constraint failed')
    return value


def parse_json(text, kind):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'{kind}: duplicate field {key}')
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{kind}: invalid JSON at line {exc.lineno}') from None
    return value


def load_json(path, kind):
    return parse_json(path.read_text(), kind)


def load(path, kind):
    return validate(load_json(path, kind), kind)
