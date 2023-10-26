import glob
import argparse
import numpy as np
import cv2

def DICE(Vref,Vseg):
	dice=2*(Vref & Vseg).sum()/(Vref.sum() + Vseg.sum())
	return dice

def png_series_reader(dir):
	V = []
	png_file_list=glob.glob(dir+'/*.png')
	png_file_list.sort()
	for filename in png_file_list: 
		image = cv2.imread(filename,0)
		V.append(image)
	V = np.array(V,order='A')
	V = V.astype(bool)
	return V
# ======= Directories =======
def direct(i,root_dir,mode='train'):

	s = ''
	if mode == 'train':
		ground_dir = '../../CHAOS/Train_Sets/CT/%d/Ground' % (i)
		seg_dir = root_dir  + 'CHAOS_train/%smask/%d'%(s,i)
	elif mode == 'test':
		ground_dir = '../../../data/CHAOS_png/Test/mask/%d'%(i)
		seg_dir = root_dir + 'CHAOS_test/%smask/%d'%(s,i)
	elif mode == 'LITS':
		ground_dir = '../../../data/LITSChallenge/label/%03d'%(i)
		seg_dir = root_dir + 'LITS/%smask/%03d'%(s,i)
	return ground_dir,seg_dir

# ======= Volume Reading =======
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str)
parser.add_argument('--mode_list', nargs = '+',type=str,default=['test','LITS'])
args = parser.parse_args()
print(args)
for mode in args.mode_list:
	Vref_all,Vseg_all = [],[]
	print(mode)
	if mode == 'train':
		num_list = [1,2,5,6,8,10,14,16,18,19,21,22,23,24,25,26,27,28,29,30] # the volume index of CHAOS dataset
	elif mode == 'test':
		num_list = [2,10,25,28] # the volume index of CHAOS dataset
	elif mode == 'LITS':
		num_list = [i for i in range(0,131)] # the volume index of LITS dataset

	for i in num_list:
		ground_dir,seg_dir = direct(i,args.root_dir,mode)
		Vref = png_series_reader(ground_dir)
		Vseg = png_series_reader(seg_dir)
		try:
			Vref_all = np.vstack((Vref_all, Vref))
			Vseg_all = np.vstack((Vseg_all, Vseg))
		except:
			Vref_all = Vref
			Vseg_all = Vseg
		dice = DICE(Vref,Vseg)
		print(i,dice)

	print(Vref_all.shape)
	dice = DICE(Vref_all,Vseg_all)
	print(dice)
