from tools.metrics import evaluation

gt = 'data/labels/vid1'
pred = 'data/predictions/vid2'

evaluation(gt, pred, skip=True)