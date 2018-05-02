import os
import time
import os.path as op
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import LatentDirichletAllocation

from nilearn.datasets import fetch_atlas_aal
from nilearn import plotting
from nilearn.image import index_img, concat_imgs
from nilearn.input_data import NiftiMasker
from nilearn import plotting
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker

import joblib
import re

rng = np.random.RandomState(42)

DECONF = True
RES_NAME = 'ibp_lda' + ('_SITEDECONF' if DECONF else '')

ROI_DIR = 'DMN_subregions_extra'
roi_paths = glob.glob(ROI_DIR + '/*.nii')
roi_names = [r.split(os.sep)[-1].split('.nii')[0] for r in roi_paths]
tmp_img = nib.load('colin.nii')

roi_coords = []
for i_roi, roi in enumerate(roi_paths):
    roi_nii = nib.load(roi)
    roi_th = nib.Nifti1Image(
        np.array(roi_nii.get_data() > 0, dtype=np.int16),
        roi_nii.get_affine(),
        header=roi_nii.get_header()
    )
    rroi = resample_img(
        roi_th,
        target_affine=tmp_img.get_affine(),
        target_shape=tmp_img.shape[:3],
        interpolation='nearest'
    )
    
    cur_roi_img = nib.Nifti1Image(
        np.array(np.squeeze(rroi.get_data()) > 0, dtype=np.int32),
        affine=tmp_img.get_affine())
    roi_coords.append(plotting.find_xyz_cut_coords(cur_roi_img))

RES_NAME = 'net_pred_combined_clf_' + ROI_DIR
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

##############################################################################
# load+preprocess data
##############################################################################

print('Loading ADHD data (1=ADHD)...')
rs_files = glob.glob('/Volumes/porsche/adhd_niak/fmri*/*.nii.gz')
sub_ids = [int(f.split('_session')[0].split('X_')[-1]) for f in rs_files]

fo = open("adhd200_all.tsv", "rw+")
lines = fo.readlines()
header = lines[0]
lines = lines[1:]
subid2diag = {int(l.split('\t')[0]) : l.split('\t')[5]  for l in lines}

sub_diags = []
for sub_id, path in zip(sub_ids, rs_files):
    cur_sub_id = int(subid2diag[sub_id])
    sub_diags.append(cur_sub_id)
sub_diags = np.array(sub_diags)
sub_diags_bin = np.array(sub_diags > 0, np.int32)
    
assert len(sub_diags) == len(rs_files) == len(sub_ids)

n_subjects = len(sub_diags)
n_patients = n_subjects - np.sum(sub_diags > 0)
print('%i patients/%i total' % (n_patients, n_subjects))

labels_adhdfull = np.array(sub_diags)
labels_adhd = np.array(sub_diags_bin)
niipaths_adhd = np.array(rs_files)
sub_ids_adhd = np.array(sub_ids)


print('Loading data (2=AUTISM)...')
rs_files = glob.glob('/Volumes/porsche/abide_niak/Outputs/niak/filt_global/func_preproc/*.nii.gz')
sub_ids = [int(f.split('_func')[0].split('_')[-1]) for f in rs_files]

f = open('ABIDE.csv','r')
header = f.readline()
lines = f.readlines()

subid2diag = {int(l.split(';')[1]) : int(l.split(';')[2])  for l in lines}

sub_diags = np.array([subid2diag[sid] for sid in sub_ids])
    
assert len(sub_diags) == len(rs_files) == len(sub_ids)

labels_autism = np.array(sub_diags)
labels_autism[labels_autism == 2] = 0  # rename the healthy individuals towards 0
labels_autism[labels_autism == 1] = 2  # rename the autistic individuals towards 2

labels_autism_fulladhd = np.array(labels_autism)
labels_autism_fulladhd[labels_autism_fulladhd == 2] = 4

niipaths_autism = np.array(rs_files)
sub_ids_autism = np.array(sub_ids)

nsamples = len(labels_autism)
print "total RS images: %i" % nsamples
print "autist/control: %i/%i" % (np.sum(labels_autism==2),np.sum(labels_autism==0))

# assure there is not subject ID twice in both datasets
assert not np.any(np.in1d(sub_ids_adhd, sub_ids_autism))

niipaths_all = np.hstack((niipaths_adhd, niipaths_autism))
labels_all = np.hstack((labels_adhd, labels_autism))
labels_all_fulladhd = np.hstack((labels_adhdfull, labels_autism_fulladhd))
sub_ids_all = np.hstack((sub_ids_adhd, sub_ids_autism))

assert len(niipaths_all) == len(labels_all) == len(sub_ids_all)

assert len(labels_all[labels_all == 2]) == len(labels_all_fulladhd[labels_all_fulladhd == 4])


##############################################################################
# Construct region-region correlation feature space for each participant
##############################################################################

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)


from nilearn._utils import check_niimg
from matplotlib import pylab as plt

cross_corrs = []
rs_img_counts = []
cur_FS = np.zeros((len(niipaths_all), 210))
if op.exists('dump_FS_corrs.npy'):
    pre_FS = np.load('dump_FS_corrs.npy')
    cur_FS = pre_FS[:len(cur_FS)]
for i_rs, rs_base in enumerate(niipaths_all[:]):
    rs_4D = nib.load(rs_base)
    print('%i %s: %i images found...' % (i_rs + 1, rs_base, rs_4D.shape[-1]))
    rs_img_counts.append(rs_4D.shape[-1])
    
    
    tmp_img = index_img(rs_4D, 0)
    
    roi_paths = glob.glob(ROI_DIR + '/*.nii')
    roi_names = [r.split(os.sep)[-1].split('.nii')[0] for r in roi_paths]

    rroi_img = np.zeros(tmp_img.get_data().shape[:3])
    for i_roi, roi in enumerate(roi_paths):
        roi_nii = nib.load(roi)
        roi_th = nib.Nifti1Image(
            np.array(roi_nii.get_data() > 0, dtype=np.int16),
            roi_nii.get_affine(),
            header=roi_nii.get_header()
        )
        rroi = resample_img(
            roi_th,
            target_affine=tmp_img.get_affine(),
            target_shape=tmp_img.shape[:3],
            interpolation='nearest'
        )
        
        rroi_img[np.squeeze(rroi.get_data()) > 0] = i_roi + 1

    rroi_img = nib.Nifti1Image(
        rroi_img, affine=tmp_img.get_affine()
    )
    rroi_img.to_filename('dbg_roi_labelimg.nii.gz')
    n_rois = len(np.unique(rroi_img.get_data())) - 1
    # !fslview dbg_roi_labelimg.nii.gz

    roi_masker = NiftiLabelsMasker(
        labels_img=rroi_img,
        background_label=0,
        standardize=True,
        smoothing_fwhm=0,
        detrend=True,
        low_pass=None,
        high_pass=None,
        memory='nilearn_cache', verbose=0)
    roi_masker.fit()
    
    # parse the RS images according to the ROIs
    sub_FS = roi_masker.transform(rs_4D)
    
    # assert sub_FS.shape == (len(cur_rs_imgs), len(roi_names))
    # dump_path = op.join(rs_base, 'roi_dump_' + ROI_DIR)
    # np.save(dump_path, arr=sub_FS)

    # vectorize
    cross_corr = np.corrcoef(sub_FS.T)
    tril_inds = np.tril_indices_from(cross_corr, k=-1)
    cc_ravel = cross_corr[tril_inds]
    
    # cross_corrs.append(cc_ravel)
    cur_FS[i_rs] = cc_ravel

    # plt.matshow(cross_corr, cmap=plt.cm.RdBu)
    # plt.title(rs_base)
    # plt.yticks(np.arange(len(roi_names)), roi_names)
    # plt.show()
    # plt.colorbar()
    
    if i_rs % 10 == 0:
        np.save('dump_FS_corrs', arr=cur_FS)


# dump the engineered feature space and corresponding meta-info
cur_FS = np.nan_to_num(cur_FS)
joblib.dump((cur_FS, niipaths_all, labels_all, sub_ids_all),
            'all_FS_paths_labels_subids', compress=9)



stuff = joblib.load('all_FS_paths_labels_subids')
FS, paths, labels, sub_ids = stuff

# trim cohorts by age and by gender (only males)
import pandas
meta_autism = pandas.read_excel('Finalinput_ABIDE_ASD_TD_MALE_BELOW21.xlsx')
meta_adhd = pandas.read_excel('Finalinput_ADHD200_ADHD_TD_MALE_BELOW21.xlsx')

included_autism = meta_autism.values[:, 0]
included_adhd = meta_adhd.values[:, 0]
included_all = np.hstack((included_autism, included_adhd))

presel = np.in1d(sub_ids, included_all)
FS, paths, labels, sub_ids = FS[presel], paths[presel], labels[presel], sub_ids[presel]

rs_img_counts = np.array(rs_img_counts)[presel]
labels_all_fulladhd = labels_all_fulladhd[presel]

assert np.where(np.where(labels_all_fulladhd == 4)[0] == np.where(labels == 2)[0])
assert np.where(np.where(labels_all_fulladhd != 4)[0] != np.where(labels == 2)[0])

hc_inds = np.where(labels == 0)[0]
adhd_inds = np.where(labels == 1)[0]
autism_inds = np.where(labels == 2)[0]
mean_patients = (len(adhd_inds) + len(autism_inds)) / 2
rng.shuffle(hc_inds)
keep_inds = np.hstack((hc_inds[:mean_patients], adhd_inds, autism_inds))
FS, paths, labels, sub_ids = FS[keep_inds], paths[keep_inds], labels[keep_inds], sub_ids[keep_inds]
labels_all_fulladhd = labels_all_fulladhd[keep_inds]

joblib.dump((FS, paths, labels, sub_ids),
            'all_FS_paths_labels_subids_presel', compress=9)
joblib.dump(labels_all_fulladhd, 'all_FS_labels_all_fulladhd', compress=9)


# FS_ss = StandardScaler().fit_transform(FS)
FS_ss = FS
categ_ind = labels

# 1 -> 54
rng = np.random.RandomState(1)

clf = LinearSVC(multi_class='ovr')
folder = StratifiedShuffleSplit(categ_ind, 100, test_size=0.05, random_state=rng)
accs = []

folder_count = 0
coef = None
prfs = []
# TAGS = roi_names
# prec = np.empty((len(TAGS), folder.n_iter))
# reca = np.empty((len(TAGS), folder.n_iter))
# f1 = np.empty((len(TAGS), folder.n_iter))

from sklearn.metrics import confusion_matrix

for (train_inds, test_inds) in folder:
    clf.fit(FS_ss[train_inds], categ_ind[train_inds])

    pred = clf.predict(FS_ss[test_inds])
    cm = confusion_matrix(pred, categ_ind[test_inds])
    prfs = precision_recall_fscore_support(pred, categ_ind[test_inds])

    # prec[:, folder_count] = np.array(prfs[0])
    # reca[:, folder_count] = np.array(prfs[1])
    # f1[:, folder_count] = np.array(prfs[2])

    acc = clf.score(FS_ss[test_inds], categ_ind[test_inds])
    print acc
    print cm
    accs.append(acc)
    if coef is None:
        coef = np.zeros_like(clf.coef_)
    coef = coef + clf.coef_
    folder_count = folder_count + 1
print('---> mean acc')
acc_mean = np.mean(accs)
print (acc_mean)
print('---> mean coef')
coef = coef/folder_count


for diag, cur_coef in zip(['healthy', 'ADHD', 'Autism'], coef):
    out_coef = np.zeros((n_rois, n_rois))
    tril_inds = np.tril_indices_from(out_coef, k=-1)
    out_coef[tril_inds] = cur_coef

    from matplotlib import pylab as plt
    plt.close('all')
    plt.figure()
    # plt.imshow(np.zeros_like(out_coef), cmap=plt.cm.RdBu)
    masked_data = np.ma.masked_where(out_coef == 0., out_coef)
    plt.imshow(masked_data, cmap=plt.cm.RdBu, interpolation='nearest',
               vmin=-1, vmax=1)
    plt.xticks(np.arange(len(TAGS)), TAGS, rotation=90)
    plt.yticks(np.arange(len(TAGS)), TAGS)
    plt.title('Three-class prediction (%.0f%%): specific weights for %s' % (acc_mean * 100, diag))
    plt.colorbar()
    plt.tight_layout()
    outdir = op.join(WRITE_DIR, ROI_DIR + '_rscorr_diagnosis_%s.png' % diag)
    plt.savefig(outdir)
    plt.show()


##############################################################################
# IBP + LDA
##############################################################################

print('Loading data...')

stuff = joblib.load('all_FS_paths_labels_subids_presel')
# stuff = joblib.load('all_FSaal_paths_labels_subids_presel')
X_task, paths, sub_diags, sub_ids = stuff

# site confound regressor
import pandas as pd
from nilearn.signal import clean
if DECONF:
    print('Site deconfounding !')
    site_strs = np.array([path.split('/')[-1].split('_')[0] for path in paths])
    site_regrs = pd.get_dummies(site_strs).values
    X_task = clean(np.array(X_task), confounds=[site_regrs], standardize=False,
                   detrend=False)

n_subjects = len(sub_diags)
n_healthy = n_subjects - np.sum(sub_diags > 0)
print('%i healthy/%i total' % (n_healthy, n_subjects))

labels = np.int32(sub_diags)

# type conversion
X_task = np.float32(X_task)
labels = np.int32(labels)

X_task = StandardScaler().fit_transform(X_task)


for i_iter in np.arange(70):
    print i_iter

    alpha = rng.randint(1, 20)
    n_topics = 3

    # n_topics = np.random.randint(2, 6)
    ROI_TAG = 'DMNsubregionsextra'
    # ROI_TAG = 'AAL'

    n_feat = X_task.shape[-1]


    # NBP

    from ibp.ugs import UncollapsedGibbsSampling

    # set up the hyper-parameter for sampling alpha
    alpha_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_x
    sigma_x_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_a
    sigma_a_hyper_parameter = (1., 1.);

    # features = features.astype(numpy.int);

    # initialize the model
    ibp_exp_dict = {}

    print('-' * 80)
    print alpha
    try:
        ibp = UncollapsedGibbsSampling(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);
        # ibp._initialize(data[1:500, :], 1.0, 0.2, 0.5, None, None, None);
        ibp._initialize(
            data=X_task,
            alpha=alpha,  # the higher, the more clusters
            sigma_f=0.2,
            sigma_x=0.5,
            initial_Z=None,
            A_prior=None,  #features[0, :],
            initial_A=None);
        ibp.sample(30);  # execute !
    except:
        continue

    IBP_weights = ibp._Z
    IBP_clusts = len(ibp._A)
    IBP_comps = ibp._A

    proj_X_bin = IBP_weights



    # LDA

    print('Binarization: %i zeros / %i others' % (
        (proj_X_bin == 0).sum(), (proj_X_bin != 0).sum()
    ))

    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=30,
                                    learning_method='online', learning_offset=50.,
                                    random_state=rng)

    lda.fit(proj_X_bin[sub_diags != 0])
    topic_weights = lda.transform(proj_X_bin)

    n_feat = IBP_clusts


    # classify on projected features

    from scipy.stats import pearsonr
    import pandas

    adhd_meta = pandas.read_excel('adhd200_all_corr.xls')
    autism_meta = pandas.read_excel('ABIDE.xlsx')
    f = open('ABIDE.csv','r')
    autism_columns = np.array(f.readline().split(';'))


    PREF = 'ibplda_it%i_' % i_iter

    diag_names = np.array(['HC', 'ADHD', 'Autism'])
    for idiag in [1, 2]: #np.unique(sub_diags):
        group_inds = sub_diags == idiag
        group_subids = sub_ids[group_inds]
        
        group_weights = topic_weights[group_inds]
        
        # correlate latent factors with the behavioral meta data
        if diag_names[idiag] == 'ADHD':
            our_order = np.array([int(np.where(adhd_meta.values[:, 0] == cur_id)[0]) for cur_id in group_subids])
            group_meta = adhd_meta.values[our_order]
            
            i_cols = np.in1d(adhd_meta.columns, np.array(['ADHD Measure', 'DX', 'ADHD Index', 'Inattentive', 'Hyper/Impulsive', 'Verbal IQ', 'Performance IQ', 'Full4 IQ', 'Med Status']))
            adhd_col_names = adhd_meta.columns.values[i_cols]
            r_abs = []
            infos = []
            list_colnames = []
            adhd_mat_corrs = np.zeros((len(adhd_col_names), n_topics))
            for col, name in zip(group_meta[:, i_cols].T, adhd_col_names):
                i_nans = np.array([np.isnan(val) for val in col])
                col[i_nans] = 0
                i_999 = np.array([val == -999 for val in col])
                col[i_999] = 0
                
                # col = np.ma.masked_where(np.logical_or(i_nans, i_999), col)
                i_good = np.logical_not(np.logical_or(i_nans, i_999))
                
                for i_top in range(n_topics):
                    r, p = pearsonr(col[i_good], topic_weights[group_inds, i_top][i_good])
                    info = 'ADHD: Corr factor%i~%s: %.2f' % (i_top + 1, name, r)
                    adhd_mat_corrs[adhd_col_names == name, i_top] = r
                    r_abs.append(np.abs(r))
                    infos.append(info)
            r_abs = np.array(r_abs)
            cool_sort = np.argsort(r_abs)[::-1]
            for i_ranked in cool_sort:
                print(infos[i_ranked])

        if diag_names[idiag] == 'Autism':
            garbage_offset = 7
            autism_values = autism_meta.values[garbage_offset:]
            our_order = np.array([int(np.where(autism_values[:, 1] == cur_id)[0]) for cur_id in group_subids])
            group_meta = autism_values[our_order]
            
            i_cols = np.in1d(autism_columns, np.array(['DSM_IV_TR', 'ADOS_MODULE', 'ADOS_TOTAL', 'ADOS_COMM', 'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV']))
            autism_col_names = autism_columns[i_cols]
            r_abs = []
            infos = []
            autism_mat_corrs = np.zeros((len(autism_col_names), n_topics))
            for col, name in zip(group_meta[:, i_cols].T, autism_col_names):
                i_nans = np.array([np.isnan(val) for val in col])
                col[i_nans] = 0
                i_999 = np.array([val == -9999 for val in col])
                col[i_999] = 0
                
                # col = np.ma.masked_where(np.logical_or(i_nans, i_999), col)
                i_good = np.logical_not(np.logical_or(i_nans, i_999))
                
                for i_top in range(n_topics):
                    r, p = pearsonr(col[i_good], topic_weights[group_inds, i_top][i_good])
                    info = 'Autism: Corr factor%i~%s: %.2f' % (i_top + 1, name, r)
                    autism_mat_corrs[autism_col_names == name, i_top] = r
                    r_abs.append(np.abs(r))
                    infos.append(info)
            r_abs = np.array(r_abs)
            cool_sort = np.argsort(r_abs)[::-1]
            for i_ranked in cool_sort:
                print(infos[i_ranked])


    # plotting
    ROI_DIR = 'DMN_subregions_extra'
    roi_paths = glob.glob(ROI_DIR + '/*.nii')
    TAGS = [r.split(os.sep)[-1].split('.nii')[0] for r in roi_paths]
    MAX_THRESH = 0.25

    WRITE_DIR = op.join(os.getcwd(), RES_NAME)
    WRITE_DIR += '_%s' % ROI_TAG
    WRITE_DIR += '_top%i' % n_topics
    WRITE_DIR += '_prop%i' % IBP_clusts
    WRITE_DIR += '_corr%.4f' % ((np.abs(adhd_mat_corrs).sum() + np.abs(autism_mat_corrs).sum()) / n_topics)
    if not op.exists(WRITE_DIR):
        os.mkdir(WRITE_DIR)

    plt.close('all')
    TH = 0.35
    for group in ['ADHD', 'Autism']:
        sns.set(style="white", context="talk")
        f, (axes) = plt.subplots(n_topics, 1, figsize=(8, 9), sharex=True)
        for i_topic in np.arange(n_topics):
            if 'ADHD' in group:
                col_names = adhd_col_names
                mat_corrs = adhd_mat_corrs
            else:
                col_names = autism_col_names
                mat_corrs = autism_mat_corrs
            color_order = np.array(["#e74c3c"] * len(mat_corrs[:, i_topic]))
            color_order[mat_corrs[:, i_topic] < 0] = "#3498db"
            my_palette = sns.color_palette(color_order)
            bar_hdl = sns.barplot(col_names, mat_corrs[:, i_topic], ax=axes[i_topic],
                                  palette=my_palette)
            for item in bar_hdl.get_xticklabels():
                item.set_rotation(45)
            for i_r, p in enumerate(bar_hdl.patches):
                r = mat_corrs[i_r, i_topic]
                if r > 0:
                    height = p.get_y() + p.get_height() + 0.08
                else:
                    height = p.get_y() - p.get_height() - 0.12
                bar_hdl.text(p.get_x() + p.get_width() / 6., height, '%0.2f' % r)
            axes[i_topic].set_ylabel("Factor %i" % (i_topic + 1))
        sns.despine(bottom=True)
        plt.suptitle('Latent patterns in %s' % group)
        plt.setp(f.axes, yticks=[-TH, 0, TH])
        plt.tight_layout(h_pad=3)
        outdir = op.join(WRITE_DIR, PREF + '_corrs_%s.png' % group)
        plt.savefig(outdir)
        plt.show()

    factor_weights = np.dot(lda.components_, IBP_comps)
    for i_c, comp_mat in enumerate(factor_weights):
        if 'AAL' in ROI_TAG:
            aal_ds = fetch_atlas_aal()
            TAGS = aal_ds['labels']
            out_coef = np.zeros((116, 116))
        else:
            out_coef = np.zeros((21, 21))
        tril_inds = np.tril_indices_from(out_coef, k=-1)
        out_coef[tril_inds] = comp_mat
    
        from matplotlib import pylab as plt
        if 'AAL' in ROI_TAG:
            plt.figure(figsize=(14, 12))
        else:
            plt.figure(figsize=(10, 8))
        # plt.imshow(np.zeros_like(out_coef), cmap=plt.cm.RdBu)
        masked_data = np.ma.masked_where(out_coef == 0., out_coef)
        # plt.imshow(masked_data, cmap=plt.cm.RdBu, interpolation='nearest')

        # if (masked_data < 0).sum() == 0:
        #     my_cmap = plt.cm.Reds
        # else:
        #     my_cmap = plt.cm.RdBu_r
        my_cmap = plt.cm.RdBu_r
        plt.imshow(masked_data, cmap=my_cmap, interpolation='nearest',
                #    vmin=-40, vmax=40
                   )
        plt.colorbar()
        # heat_hdl = sns.heatmap(masked_data, xticklabels=10, yticklabels=10, cbar_kws={"shrink": .5}, cmap=my_cmap)
        # for item in heat_hdl.get_yticklabels():
        #     item.set_rotation(5)
        if 'AAL' in ROI_TAG:
            plt.xticks(np.arange(len(TAGS)), TAGS, fontsize=5, rotation=90)
            plt.yticks(np.arange(len(TAGS)), TAGS, fontsize=5)
        else:
            plt.xticks(np.arange(len(TAGS)), TAGS, fontsize=12, rotation=90)
            plt.yticks(np.arange(len(TAGS)), TAGS, fontsize=12)
        plt.title('Factor %i' % (i_c + 1), fontsize=19)
        plt.grid(False)
        plt.tight_layout()
    
        outdir = op.join(WRITE_DIR, 'factor%i-%i.png' % (i_c + 1, len(factor_weights)))
        plt.savefig(outdir)
        plt.show()

        # glassbrain
        def symmetrize(a):
            return a + a.T - np.diag(a.diagonal())

        title = ""
        f = plt.figure(figsize=(16, 8))
        mat = symmetrize(out_coef)
        rat = 1 / np.max([np.max(mat), np.abs(np.min(mat))])
        mat *= rat
        plotting.plot_connectome(mat, roi_coords, figure=f, edge_cmap=plt.cm.RdBu_r,
                                 edge_threshold='0%', display_mode="lzry",
                                 node_size=150, colorbar=True,
                                 title=title, #node_color=['grey'] * 21,
                                 #edge_vmin=-0.5, edge_vmax=0.3
                                 )
        outdir = op.join(WRITE_DIR, 'glassbrain_factor%i-%i.png' % (i_c + 1, len(factor_weights)))
        plt.savefig(outdir)
        plt.show()                  

    plt.figure()
    data = IBP_comps
    sns.heatmap(data, xticklabels=10, yticklabels=10, cbar_kws={"shrink": .5})
    plt.xticks((np.arange(data.shape[1]))[::10], (np.arange(data.shape[1]) + 1)[::10])
    plt.yticks(np.arange(data.shape[0])[::5], (np.arange(data.shape[0]) + 1)[::5])
    plt.xlabel('Seed-seed correlation features')
    plt.ylabel('Hidden properties')
    plt.title('IBP: Latent property discovery')
    outdir = op.join(WRITE_DIR, PREF + '_ibp.png')
    plt.savefig(outdir)
    plt.show()


    plt.figure()
    sns.heatmap(lda.components_, xticklabels=10, yticklabels=n_topics, cbar_kws={"shrink": .5})
    plt.xticks((np.arange(n_feat))[::5], (np.arange(n_feat) + 1)[::5])
    plt.yticks(np.arange(n_topics), np.arange(n_topics) + 1)
    plt.ylabel('LDA components')
    plt.xlabel('Latent properties from IBP')
    plt.title('LDA: weights')
    outdir = op.join(WRITE_DIR, PREF + '_lda.png')
    plt.savefig(outdir)
    plt.show()
    
    joblib.dump(ibp, WRITE_DIR + '/dump_ibp', compress=9)
    joblib.dump(lda, WRITE_DIR + '/dump_lda', compress=9)


from sklearn.manifold import TSNE

wrapper = TSNE(n_components=3)
Y = wrapper.fit_transform(IBP_weights)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
diag_names = ['Healthy', 'ADHD', 'ASD']
for diag, color in ((0, 'b'), (1, 'r'), (2, 'g')):
    ax.scatter(Y[sub_diags==diag, 0], #[100:200],
               Y[sub_diags==diag, 1], #[100:200],
               Y[sub_diags==diag, 2], #[100:200],
               c=color, marker='o', label=diag_names[diag])

min_th = -Y.std() * 2
max_th = +Y.std() * 2
ax.set_xlim(min_th, max_th)
ax.set_ylim(min_th, max_th)
ax.set_zlim(min_th, max_th)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.legend(fontsize='large', loc='upper left')
plt.show()

plt.savefig('cool_3D_plot_end1.png', dpi=500)


##############################################################################
# Comparison to clinical labels
##############################################################################

print('Loading data...')

sub_diags_fulladhd = joblib.load('all_FS_labels_all_fulladhd')

n_subjects = len(sub_diags)
n_patients = n_subjects - np.sum(sub_diags > 0)
print('%i patients/%i total' % (n_patients, n_subjects))

# X_task = StandardScaler().fit_transform(X_task)
# labels = np.int32(sub_diags > 0)
labels = np.int32(sub_diags)

# type conversion
X_task = np.float32(X_task)
labels = np.int32(labels)


IBP_weights = ibp._Z
IBP_clusts = len(ibp._A)
IBP_comps = ibp._A
topic_weights = lda.transform(IBP_weights)



"""
Derive pre-validation topic weights and labels
"""
cond_labels = np.array(labels)
folder = StratifiedKFold(y=cond_labels, n_folds=10, random_state=42)
clf = LinearSVC(penalty='l2')
n_topics = 3
prev_labels = np.zeros((IBP_weights.shape[0], n_topics))

real_acc = {}
name = 'internal_model'
for train_inds, test_inds in folder:
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=30,
                                    learning_method='online', learning_offset=50.,
                                    random_state=42)
    lda.fit(IBP_weights[train_inds])
    in_topicweights = lda.transform(IBP_weights[train_inds])
    out_topicweights = lda.transform(IBP_weights[test_inds])
    
    prev_labels[test_inds, :] = out_topicweights

    clf.fit(in_topicweights, cond_labels[train_inds])
    pred_labels = clf.predict(out_topicweights)

    cur_acc = np.mean(cond_labels[test_inds] == pred_labels)
    if name in real_acc.keys():
        real_acc[name].append(cur_acc)
    else:
        real_acc[name] = [cur_acc]
print('Real mean accuracy on %s: %.2f +/-%.2f' % (name, np.mean(real_acc[name]) * 100, np.std(real_acc[name]) * 100))

# build binary vector from continuous topic weights
prev_labels_categ = np.argmax(prev_labels, axis=1)
prev_labels_bin = np.zeros_like(prev_labels)
for i, cur_ind in enumerate(prev_labels_categ):
    prev_labels_bin[i, cur_ind] = 1


# conclusion: 67% acc on prevalidated topic labels (=integer) versus 48% on original disease labels
real_acc = {}
for X, name in zip([X_task, X_task], ['B Factor labels', 'A Original labels']):
    if 'Factor' in name:
        cond_labels = np.array(prev_labels_categ)
    else:
        cond_labels = np.array(labels)
    folder = StratifiedKFold(y=cond_labels, n_folds=10, random_state=42)
    clf = LinearSVC(penalty='l2')
    for train_inds, test_inds in folder:
        clf.fit(X[train_inds], cond_labels[train_inds])
        pred_labels = clf.predict(X[test_inds])
        
        cur_acc = np.mean(cond_labels[test_inds] == pred_labels)
        if name in real_acc.keys():
            real_acc[name].append(cur_acc)
        else:
            real_acc[name] = [cur_acc]
    print('Out-of-sample accuracy on %s: %.2f +/-%.2f' % (name, np.mean(real_acc[name]) * 100, np.std(real_acc[name]) * 100))

k = real_acc.keys()
t, p = ttest_rel(real_acc[k[0]], real_acc[k[1]])
sign_tag = 'Difference is significant (p=%.4f)' % p

import pandas as pd
plt.figure()
df_accs = pd.DataFrame(real_acc)
ax = sns.violinplot(x=None, y=None, data=df_accs, palette="muted")
plt.title('Comparing classification accuracy of clinical and biological labels', fontsize=15)
plt.ylim(0.2, 0.8)
plt.ylabel('Out-of-sample performance')
plt.xlabel(sign_tag)
plt.axhline(y=0.33, linewidth=1, color='r')
plt.savefig(WRITE_DIR + '/acc_org_vs_topic_labels.png')

inital_org_accs = real_acc['A Original labels']


# conclusio: Smaller model variance but same prediction acc when adding prevalidated topic weights
real_acc = {}
X_task_plus_topicweights = np.hstack((X_task, prev_labels))
for X, name in zip([X_task_plus_topicweights, X_task], ['B Original features + Factor weights', 'A Original features']):
    folder = StratifiedKFold(y=cond_labels, n_folds=10, random_state=1)
    clf = LinearSVC(penalty='l2')
    for train_inds, test_inds in folder:
        clf.fit(X[train_inds], cond_labels[train_inds])
        pred_labels = clf.predict(X[test_inds])
        
        cur_acc = np.mean(cond_labels[test_inds] == pred_labels)
        if name in real_acc.keys():
            real_acc[name].append(cur_acc)
        else:
            real_acc[name] = [cur_acc]
    print('Out-of-sample accuracy on %s: %.2f +/-%.2f' % (name, np.mean(real_acc[name]) * 100, np.std(real_acc[name]) * 100))

real_acc['A Original features'] = inital_org_accs
print df_accs

k = real_acc.keys()
t, p = ttest_rel(real_acc[k[0]], real_acc[k[1]])
sign_tag = 'Difference is not significant (p=%.4f)' % p

import pandas as pd
plt.figure()
df_accs = pd.DataFrame(real_acc)
ax = sns.violinplot(x=None, y=None, data=df_accs, palette="muted")
plt.title('Comparing original features and augmentation with factor weights', fontsize=15)
plt.ylabel('Out-of-sample performance')
plt.xlabel(sign_tag)
plt.ylim(0.2, 0.8)
plt.axhline(y=0.33, linewidth=1, color='r')
plt.savefig(WRITE_DIR + '/acc_org_vs_augmented_fs.png')


# Pseudo Pre-Validation
# Conclusio:
# As shown by classical dependent Student T-tests, the candidate endophenotypes
# inferred in unseen data are non-inferior to whole connectivity matrix
# in predicting the clinical diagnoses HC - ADHD - Autism.
# Reducting from 210 seed-seed connectivity values to 3 phenotype indicators
# yields comparable predictive accuracy.
from scipy.stats import ttest_rel
cond_labels = np.array(labels)
clf = LinearSVC(penalty='l2')
real_acc = {}
for X, name in zip([prev_labels, X_task], ['B Factor weights', 'A Original features']):
    folder = StratifiedKFold(y=cond_labels, n_folds=10, random_state=42)
    for train_inds, test_inds in folder:
        clf.fit(X[train_inds], cond_labels[train_inds])
        pred_labels = clf.predict(X[test_inds])

        cur_acc = np.mean(cond_labels[test_inds] == pred_labels)
        if name in real_acc.keys():
            real_acc[name].append(cur_acc)
        else:
            real_acc[name] = [cur_acc]
    print('Real mean accuracy on %s: %.2f +/-%.2f' % (name, np.mean(real_acc[name]) * 100, np.std(real_acc[name]) * 100))
k = real_acc.keys()
t, p = ttest_rel(real_acc[k[0]], real_acc[k[1]])
sign_tag = 'Difference is not significant (p=%.4f)' % p

real_acc['A Original features'] = inital_org_accs

df_accs = pd.DataFrame(real_acc)
print df_accs
plt.figure()
ax = sns.violinplot(x=None, y=None, data=df_accs, palette="muted")
plt.title('Comparing original connectivity features (%i) and only factor weights (k=%i)' % (X_task.shape[-1], topic_weights.shape[-1]), fontsize=15)
plt.ylabel('Out-of-sample performance')
plt.xlabel(sign_tag)
plt.ylim(0.2, 0.8)
plt.axhline(y=0.33, linewidth=1, color='r')
plt.savefig(WRITE_DIR + '/topic_weights_vs_org_fs.png')


# stacked barplot: distribution of latent properties in three diagnostic groups
n_props = IBP_weights.shape[-1]
prop_counts = {}
prop_counts['Healthy'] = np.sum(IBP_weights[labels == 0], axis=0)
prop_counts['ADHD'] = np.sum(IBP_weights[labels == 1], axis=0)
prop_counts['Autism'] = np.sum(IBP_weights[labels == 2], axis=0)
prop_counts['Total'] = prop_counts['Healthy'] + prop_counts['ADHD'] + prop_counts['Autism']
prop_counts['Healthy'] /= prop_counts['Total']  # normalize
prop_counts['ADHD'] /= prop_counts['Total']
prop_counts['Autism'] /= prop_counts['Total']
df_accs = pd.DataFrame(prop_counts)

col_healthy = '#dadaeb'
col_adhd = '#756daf'
col_autism = '#542c8d'


tics = np.arange(n_props) + 1
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
sns.barplot(x=tics, y=prop_counts['ADHD'] + prop_counts['Healthy'] + prop_counts['Autism'], color=col_adhd)
sns.barplot(x=tics, y=prop_counts['Healthy'] + prop_counts['Autism'], color=col_healthy)
sns.barplot(x=tics, y=prop_counts['Autism'], color=col_autism)
plt.title('Distribution of hidden connectivity properties in three clinical groups', fontsize=15)
hbar = plt.Rectangle((0,0),1,1,fc=col_healthy, edgecolor = 'none')
adhdbar = plt.Rectangle((0,0),1,1,fc=col_adhd, edgecolor = 'none')
autismbar = plt.Rectangle((0,0),1,1,fc=col_autism, edgecolor = 'none')
plt.ylabel('Assignment ratio')
plt.axhline(y=0.5, linewidth=1, color='r', linestyle='-')
plt.xlabel('Hidden property')
l = plt.legend([hbar, adhdbar, autismbar],
               ['Healthy', 'ADHD', 'Autism'], loc='upper center', ncol = 3, prop={'size':11})
l.draw_frame(True)
plt.xticks(np.arange(n_props)[::2], tics[::2])
plt.yticks(np.linspace(0, 1, 4), np.round(np.linspace(0, 1, 4), decimals=2))
plt.savefig(WRITE_DIR + '/property_assignments.png')
plt.show()

# parse the ROIs
from nilearn.image import index_img, concat_imgs
from nilearn import plotting

ROI_DIR = 'DMN_subregions_extra'
roi_paths = glob.glob(ROI_DIR + '/*.nii')
roi_names = [r.split(os.sep)[-1].split('.nii')[0] for r in roi_paths]
tmp_img = nib.load('colin.nii')

roi_coords = []
for i_roi, roi in enumerate(roi_paths):
    roi_nii = nib.load(roi)
    roi_th = nib.Nifti1Image(
        np.array(roi_nii.get_data() > 0, dtype=np.int16),
        roi_nii.get_affine(),
        header=roi_nii.get_header()
    )
    rroi = resample_img(
        roi_th,
        target_affine=tmp_img.get_affine(),
        target_shape=tmp_img.shape[:3],
        interpolation='nearest'
    )
    
    cur_roi_img = nib.Nifti1Image(
        np.array(np.squeeze(rroi.get_data()) > 0, dtype=np.int32),
        affine=tmp_img.get_affine())
    roi_coords.append(plotting.find_xyz_cut_coords(cur_roi_img))

for cur_comp in np.arange(n_props):
    if 'AAL' in ROI_DIR:
        aal_ds = fetch_atlas_aal()
        TAGS = aal_ds['labels']
        out_coef = np.zeros((116, 116))
    else:
        out_coef = np.zeros((21, 21))
    tril_inds = np.tril_indices_from(out_coef, k=-1)
    out_coef[tril_inds] = IBP_comps[cur_comp]

    def symmetrize(a):
        return a + a.T - numpy.diag(a.diagonal())

    title = "Latent property %i/%i" % (cur_comp + 1, n_props)
    f = plt.figure(figsize=(16, 8))
    mat = symmetrize(out_coef)
    rat = 1 / np.max([np.max(mat), np.abs(np.min(mat))])
    mat *= rat
    plotting.plot_connectome(mat, roi_coords, figure=f, edge_cmap=plt.cm.RdBu_r,
                             edge_threshold='0%', display_mode="lzry",
                             node_size=150, colorbar=True,
                             title=title, #node_color=['grey'] * 21,
                             #edge_vmin=-0.5, edge_vmax=0.3
                             )
    outdir = op.join(WRITE_DIR, 'glassbrain_property%i-%i.png' % (cur_comp + 1, n_props))
    plt.savefig(outdir)
    plt.show()



