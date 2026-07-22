from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from envs.delivery_env import DynamicDeliveryEnv
from evaluation.scenario_bank import load_scenario_bank, verify_scenario_hashes, sha256_json

class ScenarioBankEnvironmentFactory:
    def __init__(self, bank_manifest_path: str | Path, *, expected_split: str, expected_bank_hash: str | None = None) -> None:
        self.bank_manifest_path=Path(bank_manifest_path)
        self.bank=load_scenario_bank(self.bank_manifest_path)
        if self.bank.split != expected_split: raise ValueError(f'bank split {self.bank.split} != {expected_split}')
        if expected_split == 'test': raise ValueError('test scenario bank cannot be used for training or validation')
        if expected_bank_hash and self.bank.bank_hash != expected_bank_hash: raise ValueError('scenario bank hash mismatch')
        self._by_id={s.scenario_id:s for s in self.bank.scenarios}
        for s in self.bank.scenarios: verify_scenario_hashes(s)
    @property
    def scenario_ids(self) -> tuple[str, ...]: return tuple(self._by_id)
    @property
    def bank_hash(self) -> str: return self.bank.bank_hash
    def metadata(self, scenario_id: str) -> dict[str, Any]:
        s=self._by_id[scenario_id]; m=json.loads(Path(s.scenario_manifest_path).read_text())
        return {'scenario_id':s.scenario_id,'scenario_split':s.split,'scenario_content_hash':m.get('scenario_content_hash',s.scenario_content_hash),'instance_hash':s.instance_hash,'scenario_bank_hash':self.bank.bank_hash,'scenario_bank_path':str(self.bank_manifest_path),'scenario_manifest_hash':s.scenario_manifest_hash}
    def create(self, scenario_id: str) -> DynamicDeliveryEnv:
        if scenario_id not in self._by_id: raise KeyError(scenario_id)
        s=self._by_id[scenario_id]; verify_scenario_hashes(s)
        env=DynamicDeliveryEnv(s.instance_path)
        env.scenario_metadata=self.metadata(scenario_id)
        return env
