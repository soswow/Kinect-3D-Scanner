"""Artificial read-only baseline source for historical guard unit tests.

Actual current-source acceptance/refusal is tested separately. No scanner
objects, numerical work, report authority or production files are changed.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=Path(__file__).with_name("fixtures")/"final-source-fixture.json"
SHA="7870a140570ec4b1f334e8d5768fa826bca26012b1143aee4950bbcaefd56098"

@contextmanager
def baseline_sources():
    raw=FIXTURE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA:raise ValueError("Artificial source fixture changed")
    modules=json.loads(raw)["modules"]
    original=Path.read_text
    def reader(path,*args,**kwargs):
        text=original(path,*args,**kwargs)
        try:rel=path.resolve().relative_to(ROOT).as_posix()
        except ValueError:return text
        if rel not in modules:return text
        tree=ast.parse(text);lines=text.splitlines(keepends=True)
        nodes={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        edits=[]
        for item in modules[rel]:
            node=ast.parse(item["source"].lstrip()).body[0]
            if hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()!=item["ast_sha256"]:
                raise ValueError("Artificial function AST changed")
            current=nodes[item["name"]]
            edits.append((current.lineno-1,current.end_lineno,item["source"]))
        for start,end,source in sorted(edits,reverse=True):lines[start:end]=[source]
        return "".join(lines)
    with patch.object(Path,"read_text",reader):yield
