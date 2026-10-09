"""Distinct registered authorization for the new setup-reuse ownership method.

The complete-bridge validator alone mints permits after its own fresh audit.
No v3 loop token, old component token or truthy callback grants workspace timing.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from scripts.research import gpu_icp_device_loop_protocol as values

_REGISTRY={}


@dataclass(frozen=True)
class WorkspaceTimingPermit:
    report_path:str
    report_sha256:str
    binding_json:str
    workspace_json:str
    calls_json:str


def _mint(path,binding,workspace,calls):
    """Private seam used only by the separate complete-bridge audit validator."""
    from scripts.research import microbatch_bridge_protocol as producer
    producer._require_mint_context(path,binding,workspace,calls)
    token=WorkspaceTimingPermit(str(Path(path).resolve()),values.sha(path),values.canonical(binding),
        values.canonical(workspace),values.canonical(calls))
    _REGISTRY[id(token)]=(token,tuple(token.__dict__.values()))
    return token


def registered(token):
    record=_REGISTRY.get(id(token))
    values.require(type(token) is WorkspaceTimingPermit and record is not None and record[0] is token
        and tuple(token.__dict__.values())==record[1],"Require fresh registered workspace ownership permit, never v3/Microbatch/constructed token")
    values.require(values.sha(token.report_path)==token.report_sha256,"Closed complete-bridge audit changed")
    return token


def constructor_authority(token,workspace,configuration):
    registered(token)
    values.require(values.canonical(workspace)==token.workspace_json,"New workspace setup source/configuration was not audited")
    calls=json.loads(token.calls_json)
    values.require(calls and all(call["input"]["configuration"][key]==value for call in calls
        for key,value in configuration.items()),"Template graph/device/budget differs from all audited complete bridge calls")
    return token


def validate_start(token,index,consumed,source):
    registered(token)
    calls=json.loads(token.calls_json)
    values.require(type(index) is int and 0<=index<len(calls),"Exact ordered dynamic call suffix required")
    values.require(consumed["configuration"]["audit_nearest"] is False
        and consumed["configuration"]["audit_misses"] is False,"Timing excludes query observational audits")
    expected=calls[index]
    values.require(values.normalized_input(consumed)==expected["input"]
        and values.normalized_source(source)==expected["source"],"Actual ordered seed/arrays/graph/runtime/source were not audited for workspace timing")
    return token


def validate_terminal(token,index,consumed,source,terminal):
    validate_start(token,index,consumed,source)
    expected=json.loads(token.calls_json)[index]
    values.require(values.canonical(terminal)==values.canonical(expected["terminal"]),
        "Workspace timed terminal bytes/metrics/query/update counts changed")
    return True
