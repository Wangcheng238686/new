"""Hand-computed CPU cases for validation-only detection telemetry."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from pycocotools import mask as mu
P=Path(__file__).resolve().parents[1]/'scripts/analysis/ue_native_detection_telemetry_20261028.py'
spec=importlib.util.spec_from_file_location('detection_telemetry',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def rle(w=10,h=10):
    value=np.zeros((128,128),np.uint8);value[:h,:w]=1
    encoded=mu.encode(np.asfortranarray(value));encoded['counts']=encoded['counts'].decode();return encoded

def row(gt,pred,scores):
    return dict(image_id=1,gt=dict(bboxes=gt,labels=[0]*len(gt),rles=[rle() for _ in gt]),
        dt=dict(bboxes=pred,scores=scores,labels=[0]*len(pred)))


class TelemetryTest(unittest.TestCase):
    def test_FP_partition_and_conservation(self):
        r=row([[0,0,10,10]],[[0,0,10,10],[0,0,10,10],[5,0,15,10],[40,40,50,50]],[.9,.8,.7,.6])
        result=m.calculate([r],'dt',.05,.5)
        self.assertEqual(result['counts'],dict(images=1,GT=1,predictions_before_score=4,predictions=4,TP=1,FP=3,FN=0))
        self.assertEqual(result['FP_types'],dict(duplicate=1,localization=1,background=1))

    def test_stable_tie_and_unique_GT(self):
        tp,fp,fn,_,_=m.match_image([[0,0,10,10],[5,0,15,10]],[[2.5,0,12.5,10],[0,0,10,10]],[.8,.8],.5,.5)
        self.assertEqual(tp,[(0,0)]);self.assertEqual(fp,[(1,'duplicate')]);self.assertEqual(fn,[1])

    def test_score_filter_diagnosis_and_inclusive_threshold(self):
        r=row([[0,0,10,10]],[[0,0,10,10]],[.25])
        low=m.calculate([r],'dt',.25,.75);high=m.calculate([r],'dt',.5,.75)
        self.assertEqual(low['counts']['TP'],1);self.assertEqual(high['counts']['FN'],1)
        self.assertEqual(high['FN_best_candidate_IoU']['all_before_score']['ge_0.75'],1)
        self.assertEqual(high['FN_best_candidate_IoU']['retained_after_score']['no_candidates'],1)

    def test_bbox_error_units_and_logratios(self):
        r=row([[0,0,10,10]],[[1,0,13,10]],[.9]);result=m.calculate([r],'dt',.5,.5)
        e=result['TP_bbox_errors'];self.assertEqual(e['center_dx_px']['mean_signed'],2)
        self.assertAlmostEqual(e['center_dx_over_gt_width']['mean_abs'],.2)
        self.assertEqual(e['width_error_px']['mean_signed'],2);self.assertAlmostEqual(e['log_width_ratio']['mean_signed'],np.log(1.2))
        self.assertEqual(e['height_error_px']['mean_abs'],0)
        self.assertTrue(all(v['count']==1 for v in e.values()))
        self.assertEqual(m.calculate([r],'dt',.5,.75)['counts']['TP'],0)

    def test_empty_GT_empty_predictions_and_zero_denominators(self):
        empty=m.calculate([row([],[],[])],'dt',.5,.5)
        self.assertIsNone(empty['recall']);self.assertIsNone(empty['precision'])
        self.assertTrue(all(x['mean_signed'] is None for x in empty['TP_bbox_errors'].values()))
        background=m.calculate([row([],[[0,0,10,10]],[.8])],'dt',.05,.5)
        self.assertEqual(background['FP_types']['background'],1)
        missed=m.calculate([row([[0,0,10,10]],[],[])],'dt',.05,.5)
        self.assertEqual(missed['counts']['FN'],1);self.assertEqual(missed['recall'],0)

    def test_mask_GT_size_and_independent_FP_box_size(self):
        # GT bbox is large but mask area100; recall grouping MUST read RLE area.
        result=m.calculate([row([[0,0,128,128]],[[200,200,300,300]],[.8])],'dt',.05,.5)
        self.assertEqual(result['GT_mask_area_recall']['small']['FN'],1)
        self.assertEqual(result['FP_predicted_box_area']['large']['background'],1)
        self.assertEqual([m.size_group(a) for a in [1023,1024,9215,9216]],['small','medium','medium','large'])

    def test_overlap_bins_and_invalid_geometry(self):
        self.assertEqual([m.overlap_bin(x) for x in [None,0,.1,.5,.75]],list(m.BINS))
        with self.assertRaises(AssertionError):m.match_image([],[[0,0,0,10]],[.9],.05,.5)
        with self.assertRaises(AssertionError):m.match_image([],[[0,0,1,10]],[float('nan')],.05,.5)

if __name__=='__main__':unittest.main()
