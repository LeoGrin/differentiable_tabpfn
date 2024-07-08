import random
import time
from torch.utils.checkpoint import checkpoint

from tabpfn import TabPFNClassifier
from tabpfn.utils import normalize_data, to_ranking_low_mem, remove_outliers
from tabpfn.utils import NOP, normalize_by_used_features_f
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import check_is_fitted
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import contextlib
import torch


def preprocess_input(eval_xs, eval_ys, eval_position, device, preprocess_transform="none",
                     max_features=100, normalize_with_sqrt=False, normalize_with_test=False,
                     normalize_to_ranking=False, categorical_feats=[]):
    import warnings

    if eval_xs.shape[1] > 1:
        raise Exception("Transforms only allow one batch dim - TODO")
    if preprocess_transform != 'none':
        if preprocess_transform == 'power' or preprocess_transform == 'power_all':
            pt = PowerTransformer(standardize=True)
        elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
            pt = QuantileTransformer(output_distribution='normal')
        elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
            pt = RobustScaler(unit_variance=True)

    # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
    eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)

    # Removing empty features
    eval_xs = eval_xs[:, 0, :]
    sel = [len(torch.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
    eval_xs = eval_xs[:, sel]

    warnings.simplefilter('error')
    if preprocess_transform != 'none':
        eval_xs = eval_xs.cpu().numpy()
        feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
            range(eval_xs.shape[1])) - set(categorical_feats)
        for col in feats:
            try:
                pt.fit(eval_xs[0:eval_position, col:col + 1])
                trans = pt.transform(eval_xs[:, col:col + 1])
                # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                eval_xs[:, col:col + 1] = trans
            except:
                pass
        eval_xs = torch.tensor(eval_xs).float()
    warnings.simplefilter('default')

    eval_xs = eval_xs.unsqueeze(1)

    # TODO: Caution there is information leakage when to_ranking is used, we should not use it
    eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position) \
            if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))
    # Rescale X
    eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                            normalize_with_sqrt=normalize_with_sqrt)

    return eval_xs.to(device)


def transformer_predict(model, eval_xs, eval_ys, eval_position,
                        device='cpu',
                        max_features=100,
                        style=None,
                        inference_mode=False,
                        num_classes=2,
                        extend_features=True,
                        normalize_with_test=False,
                        normalize_to_ranking=False,
                        softmax_temperature=0.0,
                        multiclass_decoder='permutation',
                        preprocess_transform='mix',
                        categorical_feats=[],
                        feature_shift_decoder=False,
                        N_ensemble_configurations=10,
                        batch_size_inference=16,
                        differentiable_hps_as_style=False,
                        average_logits=True,
                        fp16_inference=False,
                        normalize_with_sqrt=False,
                        seed=0,
                        no_grad=True,
                        return_logits=False,
                        **kwargs):
    """

    :param model:
    :param eval_xs:
    :param eval_ys:
    :param eval_position:
    :param rescale_features:
    :param device:
    :param max_features:
    :param style:
    :param inference_mode:
    :param num_classes:
    :param extend_features:
    :param normalize_to_ranking:
    :param softmax_temperature:
    :param multiclass_decoder:
    :param preprocess_transform:
    :param categorical_feats:
    :param feature_shift_decoder:
    :param N_ensemble_configurations:
    :param average_logits:
    :param normalize_with_sqrt:
    :param metric_used:
    :return:
    """
    num_classes = len(torch.unique(eval_ys))

    def predict(eval_xs, eval_ys, used_style, softmax_temperature, return_logits):
        # Initialize results array size S, B, Classes

        # no_grad disables inference_mode, because otherwise the gradients are lost
        inference_mode_call = torch.inference_mode() if inference_mode and no_grad else NOP()
        with inference_mode_call:
            start = time.time()
            output = model(
                    (used_style.repeat(eval_xs.shape[1], 1) if used_style is not None else None, eval_xs, eval_ys.float()),
                    single_eval_pos=eval_position)[:, :, 0:num_classes]

            output = output[:, :, 0:num_classes] / torch.exp(softmax_temperature)
            if not return_logits:
                output = torch.nn.functional.softmax(output, dim=-1)
            #else:
            #    output[:, :, 1] = model((style.repeat(eval_xs.shape[1], 1) if style is not None else None, eval_xs, eval_ys.float()),
            #               single_eval_pos=eval_position)

            #    output[:, :, 1] = torch.sigmoid(output[:, :, 1]).squeeze(-1)
            #    output[:, :, 0] = 1 - output[:, :, 1]

        #print('RESULTS', eval_ys.shape, torch.unique(eval_ys, return_counts=True), output.mean(axis=0))

        return output

    def preprocess_input(eval_xs, preprocess_transform):
        import warnings

        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")
        if preprocess_transform != 'none':
            if preprocess_transform == 'power' or preprocess_transform == 'power_all':
                pt = PowerTransformer(standardize=True)
            elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
                pt = QuantileTransformer(output_distribution='normal')
            elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
                pt = RobustScaler(unit_variance=True)

        # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
        eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)

        # Removing empty features
        eval_xs = eval_xs[:, 0, :]
        sel = [len(torch.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
        eval_xs = eval_xs[:, sel]

        warnings.simplefilter('error')
        if preprocess_transform != 'none':
            eval_xs = eval_xs.cpu().numpy()
            feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
                range(eval_xs.shape[1])) - set(categorical_feats)
            for col in feats:
                try:
                    pt.fit(eval_xs[0:eval_position, col:col + 1])
                    trans = pt.transform(eval_xs[:, col:col + 1])
                    # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                    eval_xs[:, col:col + 1] = trans
                except:
                    pass
            eval_xs = torch.tensor(eval_xs).float()
        warnings.simplefilter('default')

        eval_xs = eval_xs.unsqueeze(1)

        # TODO: Caution there is information leakage when to_ranking is used, we should not use it
        eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position) \
                if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))
        # Rescale X
        eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                               normalize_with_sqrt=normalize_with_sqrt)

        return eval_xs.to(device)

    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]

    model.to(device)

    model.eval()

    import itertools
    if not differentiable_hps_as_style:
        style = None

    if style is not None:
        style = style.to(device)
        style = style.unsqueeze(0) if len(style.shape) == 1 else style
        num_styles = style.shape[0]
        softmax_temperature = softmax_temperature if softmax_temperature.shape else softmax_temperature.unsqueeze(
            0).repeat(num_styles)
    else:
        num_styles = 1
        style = None
        softmax_temperature = torch.log(torch.tensor([0.8]))

    styles_configurations = range(0, num_styles)
    def get_preprocess(i):
        if i == 0:
            return 'power_all'
#            if i == 1:
#                return 'robust_all'
        if i == 1:
            return 'none'

    preprocess_transform_configurations = ['none', 'power_all'] if preprocess_transform == 'mix' else [preprocess_transform]

    if seed is not None:
        torch.manual_seed(seed)

    feature_shift_configurations = torch.randperm(eval_xs.shape[2]) if feature_shift_decoder else [0]
    class_shift_configurations = torch.randperm(len(torch.unique(eval_ys))) if multiclass_decoder == 'permutation' else [0]

    ensemble_configurations = list(itertools.product(class_shift_configurations, feature_shift_configurations))
    #default_ensemble_config = ensemble_configurations[0]

    rng = random.Random(seed)
    rng.shuffle(ensemble_configurations)
    ensemble_configurations = list(itertools.product(ensemble_configurations, preprocess_transform_configurations, styles_configurations))
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]
    #if N_ensemble_configurations == 1:
    #    ensemble_configurations = [default_ensemble_config]

    output = None

    eval_xs_transformed = {}
    inputs, labels = [], []
    start = time.time()
    for ensemble_configuration in ensemble_configurations:
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration

        style_ = style[styles_configuration:styles_configuration+1, :] if style is not None else style
        softmax_temperature_ = softmax_temperature[styles_configuration]

        eval_xs_, eval_ys_ = eval_xs.clone(), eval_ys.clone()

        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_ = eval_xs_transformed[preprocess_transform_configuration].clone()
        else:
            eval_xs_ = preprocess_input(eval_xs_, preprocess_transform=preprocess_transform_configuration)
            if no_grad:
                eval_xs_ = eval_xs_.detach()
            eval_xs_transformed[preprocess_transform_configuration] = eval_xs_

        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_ = torch.cat([eval_xs_[..., feature_shift_configuration:],eval_xs_[..., :feature_shift_configuration]],dim=-1)

        # Extend X
        if extend_features:
            eval_xs_ = torch.cat(
                [eval_xs_,
                 torch.zeros((eval_xs_.shape[0], eval_xs_.shape[1], max_features - eval_xs_.shape[2])).to(device)], -1)
        inputs += [eval_xs_]
        labels += [eval_ys_]

    inputs = torch.cat(inputs, 1)
    inputs = torch.split(inputs, batch_size_inference, dim=1)
    labels = torch.cat(labels, 1)
    labels = torch.split(labels, batch_size_inference, dim=1)
    #print('PREPROCESSING TIME', str(time.time() - start))
    outputs = []
    start = time.time()
    for batch_input, batch_label in zip(inputs, labels):
        #preprocess_transform_ = preprocess_transform if styles_configuration % 2 == 0 else 'none'
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="None of the inputs have requires_grad=True. Gradients will be None")
            warnings.filterwarnings("ignore",
                                    message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.")
            if device == 'cpu':
                output_batch = checkpoint(predict, batch_input, batch_label, style_, softmax_temperature_, True)
            else:
                with torch.cuda.amp.autocast(enabled=fp16_inference):
                    output_batch = checkpoint(predict, batch_input, batch_label, style_, softmax_temperature_, True)
        outputs += [output_batch]
    #print('MODEL INFERENCE TIME ('+str(batch_input.device)+' vs '+device+', '+str(fp16_inference)+')', str(time.time()-start))

    outputs = torch.cat(outputs, 1)
    for i, ensemble_configuration in enumerate(ensemble_configurations):
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration
        output_ = outputs[:, i:i+1, :]
        output_ = torch.cat([output_[..., class_shift_configuration:],output_[..., :class_shift_configuration]],dim=-1)

        #output_ = predict(eval_xs, eval_ys, style_, preprocess_transform_)
        if not average_logits and not return_logits:
            # transforms every ensemble_configuration into a probability -> equal contribution of every configuration
            output_ = torch.nn.functional.softmax(output_, dim=-1)
        output = output_ if output is None else output + output_

    output = output / len(ensemble_configurations)
    if average_logits and not return_logits:
        output = torch.nn.functional.softmax(output, dim=-1)

    output = torch.transpose(output, 0, 1)

    return output

@contextlib.contextmanager
def temp_seed(seed):
    # Save the current random state
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Set the temporary seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        # Restore the previous random state
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)

def get_tabpfn_logits(model, X, y, single_eval_pos, verbose=False, use_transformer_predict=False):
    device = X.device
    if not use_transformer_predict:
        X = torch.cat((X, torch.zeros((X.shape[0], 100 - X.shape[1])).to(device)), dim=1)
        #X_2 = torch.cat((X_2, torch.zeros((X_2.shape[0], 100 - X_2.shape[1])).to(device)), dim=1)


    if verbose:
        print("single_eval_pos", single_eval_pos)

    if use_transformer_predict:
        tabpfn_input = X.reshape(X.shape[0], 1, X.shape[1])
        tabpfn_output = transformer_predict(model, tabpfn_input, y.reshape(-1, 1), eval_position=single_eval_pos, device="cuda:0", no_grad=False, preprocess_transform="none",
                                            return_logits=True, N_ensemble_configurations=1)
    else:
        # normalize data
        #X = normalize_data(X) #TODO: remove this is a test
        tabpfn_output = model((X, y), single_eval_pos=single_eval_pos)
        #tabpfn_output_2 = tabpfn((X_2, y_2), single_eval_pos=single_eval_pos)
    tabpfn_output = tabpfn_output.squeeze(0)

    return tabpfn_output



# adapted from https://github.com/yandex-research/tabular-dl-tabr/blob/75105013189c76bc4f247633c2fb856bc948e579/lib/data.py#L262
class TabrQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, noise=1e-3, random_state=None, n_quantiles=1000, subsample=1_000_000_000,
                 output_distribution="normal"):
        self.noise = noise
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution

    def fit(self, X, y=None):
        # Calculate the number of quantiles based on data size
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        # Initialize QuantileTransformer
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state
        )

        # Add noise if required
        X_modified = self._add_noise(X) if self.noise > 0 else X

        # Fit the normalizer
        normalizer.fit(X_modified)
        # show that it's fitted
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        rng = np.random.default_rng(self.random_state)
        return X + noise_std * rng.standard_normal(X.shape)



def create_animation(X_false_list, X_true, save_path, step=20):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))  # Changed axs to ax
    # set axis limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    # only keep 1 batch every 10, i.e n_step_test_factor points every n_step_test_factor * 10
    #data_for_plot = data_for_plot[::n_step_test_factor]
    #data_for_plot = data_for_plot[:n_step_test_factor * 10]

    def update(frame):
        ax.clear()
        
        X_false = X_false_list[frame]
        
        ax.scatter(X_false[:, 0], X_false[:, 1], label="False", s=10, alpha=0.2, c="red")
        ax.scatter(X_true[:, 0], X_true[:, 1], label="True", s=10, alpha=0.2, c="blue")
        ax.set_title("Training Data")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.legend()

    ani = FuncAnimation(fig, update, frames=np.arange(0, len(X_false_list), step), interval=1000, repeat=False, blit=False)

    print("animation done")

    # save the animation
    ani.save(save_path)
