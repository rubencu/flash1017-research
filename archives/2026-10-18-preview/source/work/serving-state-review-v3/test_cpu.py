"""Synthetic NPZ and rejection fixtures; no model/inference imports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location("state_audit",Path(__file__).with_name("validate_boundary.py"))
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RawStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/"fixture.npz"
        self.prefix=2668
        self.arrays={}
        metadata={}
        for key in audit.expected_keys():
            if key.endswith("left_padding") or key.endswith("lengths"):
                metadata[key]={"none":True}
                continue
            if key.startswith("prime"):
                shape,dtype=[1,1,128,512],"bfloat16"
                data=np.zeros(2*65536,np.uint8)
            elif key.endswith("slot0"):
                shape,dtype=[1],"int32"
                data=np.array([self.prefix],np.int32).view(np.uint8)
            else:
                shape,dtype=[1,3],"uint8"
                data=np.array([1,2,3],np.uint8)
            metadata[key]={"none":False,"shape":shape,"dtype":dtype,"elements":int(np.prod(shape)),"bytes":data.nbytes,"sha256":audit.byte_sha(data)}
            self.arrays[key]=data
        self.arrays["probe_logits"]=np.zeros((1,1,129280),np.float32)
        self.boundary={"layers":[{"layer":i,"offset":self.prefix,"meta_state":["deepseek_v41","3",str(audit.COMPRESSORS.get(i,0))]} for i in range(40)],"prime":{"present":True,"rings":[{"stage":i,"offset":self.prefix,"max_size":128} for i in range(3)],"expected_target_offset":self.prefix},"arrays":metadata}
        self.write()

    def write(self):
        np.savez(self.path,**self.arrays)
        self.boundary["archive"]={"path":str(self.path),"sha256":audit.sha(self.path),"bytes":self.path.stat().st_size}

    def validate(self):
        return audit.validate_archive(self.boundary,self.prefix,self.path)

    def test_valid_all_slots(self):
        values=self.validate()
        result=audit.compare_values(values,values,self.boundary["arrays"],self.boundary["arrays"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["logits_compared"],129280)
        self.assertEqual(len(values),364)

    def test_wrong_archive_hash(self):
        self.boundary["archive"]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"archive hash"): self.validate()

    def test_raw_hash_is_recomputed(self):
        self.arrays["layer20_slot1"][0]=99
        self.write()
        with self.assertRaisesRegex(ValueError,"raw state hash"): self.validate()

    def test_missing_declared_slot(self):
        del self.boundary["arrays"]["layer20_slot1"]
        with self.assertRaisesRegex(ValueError,"Missing/extra"): self.validate()

    def test_missing_saved_slot(self):
        del self.arrays["layer20_slot1"]
        self.write()
        with self.assertRaisesRegex(ValueError,"NPZ entries"): self.validate()

    def test_extra_saved_slot(self):
        self.arrays["extra"]=np.zeros(1,np.uint8)
        self.write()
        with self.assertRaisesRegex(ValueError,"NPZ entries"): self.validate()

    def test_raw_vector_dtype_required(self):
        self.arrays["layer20_slot1"]=np.ones(3,np.int8)
        self.write()
        with self.assertRaisesRegex(ValueError,"raw uint8"): self.validate()

    def test_metadata_byte_size_required(self):
        self.boundary["arrays"]["prime0_physical_keys"]["elements"]=4
        with self.assertRaisesRegex(ValueError,"dimensions/byte"): self.validate()

    def test_raw_offset_required(self):
        self.arrays["layer00_slot0"]=np.array([9],np.int32).view(np.uint8)
        self.boundary["arrays"]["layer00_slot0"]["sha256"]=audit.byte_sha(self.arrays["layer00_slot0"])
        self.write()
        with self.assertRaisesRegex(ValueError,"Raw target offset"): self.validate()

    def test_prime_metadata_required(self):
        self.boundary["prime"]["rings"][1]["offset"]-=1
        with self.assertRaisesRegex(ValueError,"ring metadata"): self.validate()

    def test_probe_full_vocabulary_required(self):
        self.arrays["probe_logits"]=np.zeros((1,1,5),np.float32)
        self.write()
        with self.assertRaisesRegex(ValueError,"full-vocabulary"): self.validate()

    def test_probe_finite_required(self):
        self.arrays["probe_logits"][0,0,12]=np.nan
        self.write()
        with self.assertRaisesRegex(ValueError,"full-vocabulary"): self.validate()

    def test_byte_difference_retained(self):
        a=self.validate();b=copy.deepcopy(a)
        b["layer39_slot1"][1]=7
        result=audit.compare_values(a,b,self.boundary["arrays"],self.boundary["arrays"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["state_differences"],[{"slot":"layer39_slot1","kind":"bytes","different_bytes":1}])

    def test_equal_numeric_signed_zero_is_not_byte_equal(self):
        a=self.validate();b=copy.deepcopy(a)
        b["probe_logits"][0,0,1]=-0.0
        result=audit.compare_values(a,b,self.boundary["arrays"],self.boundary["arrays"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["logit_different_elements"],0)

    def test_incomplete_run_rejected_before_source_or_arrays(self):
        raw=Path(self.tmp.name)/"partial.jsonl"
        raw.with_suffix(".qualification.json").write_text(json.dumps({"status":"running"}))
        with self.assertRaisesRegex(ValueError,"incomplete/failed"): audit.analyze(raw)

    def test_nonfinite_json_rejected(self):
        with self.assertRaisesRegex(ValueError,"Nonfinite JSON"): audit.loads('{"x":NaN}')

    def test_bench_records_unqualified_analysis_file_superset(self):
        audit.validate_lab_pins({"bench.py":"a"*64,"report.py":"b"*64},{"inference_files_sha256":{"flash1017/bench.py":"a"*64}})

    def test_missing_or_changed_qualified_lab_pin_rejected(self):
        for recorded in ({"report.py":"b"*64},{"bench.py":"c"*64}):
            with self.assertRaisesRegex(ValueError,"inference source pins"):
                audit.validate_lab_pins(recorded,{"inference_files_sha256":{"flash1017/bench.py":"a"*64}})


if __name__=="__main__": unittest.main()
