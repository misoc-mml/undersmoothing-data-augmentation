"""
Debiasing Generative Trees
"""

import random
import logging
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed

class BalancingGenerator():
    def __init__(self, n_estimators=1, max_depth=None, min_samples_leaf=1, split_mode='x-x', gen_mode='uni', n_components=1, cov_type='full', cat_vars=None, random_state=None, n_jobs=None, verbose=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.split_mode = split_mode
        self.gen_mode = gen_mode
        self.n_components = n_components
        self.cov_type = cov_type
        self.cat_vars = cat_vars
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.logger = logging.getLogger('models.BalancingGenerator')
    
    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        return self

    def get_params(self, deep=True):
        return {'split_mode': self.split_mode, 'gen_mode': self.gen_mode, 'n_estimators': self.n_estimators, 'n_components': self.n_components, 'cov_type': self.cov_type, 'min_samples_leaf': self.min_samples_leaf, 'max_depth': self.max_depth, 'cat_vars': self.cat_vars, 'random_state': self.random_state, 'n_jobs': self.n_jobs, 'verbose': self.verbose}

    def fit(self, X, t, y):
        self._validate_state()
        
        if self.split_mode == 'x-x':
            if t is not None:
                split_in = np.concatenate([X, t.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
            else:
                split_in = np.concatenate([X, y.reshape(-1, 1)], axis=1)
            split_out = split_in
        elif self.split_mode == 'x-t':
            split_in = X
            split_out = t.flatten()
        elif self.split_mode == 'x-y':
            if t is not None:
                split_in = np.concatenate([X, t.reshape(-1, 1)], axis=1)
            else:
                split_in = X
            split_out = y.flatten()
        else:
            raise ValueError('Unrecognised split mode.')

        if t is not None:
            all_data = np.concatenate([X, t.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        else:
            all_data = np.concatenate([X, y.reshape(-1, 1)], axis=1)

        self.splitter_obj = self._fit_splitter(split_in, split_out)
        self.generators = self._fit_subgenerators(self.splitter_obj, split_in, all_data)
        
        return self

    def sample(self, n_samples=1, eps=0.01):
        if len(self.generators) == 0:
            self.logger.info("Cannot sample data. Either 'fit' has not been called, or generators failed to model created sub-populations. Try decreasing tree's depth or the number of clusters.")
            return None

        gen_data = self._generate(self.generators, n_samples)

        if eps > 0.0:
            # Add some random noise to generated data for stronger regularisation effect
            gen_data += np.random.normal(size=gen_data.shape, loc=0.0, scale=eps)
        
        if self.cat_vars:
            self._bound_cat_vars(gen_data)
        
        return gen_data

    def bucket_size(self, X):
        estimators = self.splitter_obj.estimators_ if hasattr(self.splitter_obj, 'estimators_') else [self.splitter_obj]
        sizes = []
        n_buckets = []
        for estimator in estimators:
            bucket_sizes = Counter(estimator.apply(X)).values()
            filtered_sizes = [s for s in bucket_sizes if s > 1]
            sizes.append(np.mean(filtered_sizes))
            n_buckets.append(len(filtered_sizes))
        ret_size = np.mean(sizes).astype(int)
        ret_buckets = np.mean(n_buckets).astype(int)

        return ret_size, ret_buckets

    def _validate_state(self):
        if self.gen_mode not in ('uni', 'inv', 'obs'):
            raise ValueError(f"'{self.gen_mode}' is an unrecognised generation mode.")
        
        if self.split_mode not in ('x-x', 'x-t', 'x-y'):
            raise ValueError(f"'{self.split_mode}' is an unrecognised split mode.")

    def _fit_splitter(self, input, target=None):
        result = None

        if self.n_estimators > 1:
            result = ExtraTreesRegressor(random_state=self.random_state, n_jobs=self.n_jobs, min_samples_leaf=self.min_samples_leaf, n_estimators=self.n_estimators, max_depth=self.max_depth, bootstrap=True)
        else:
            result = DecisionTreeRegressor(random_state=self.random_state, min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth)
        
        return result.fit(input, target)
    
    def _find_best_gmm(self, data, max_comps):
        lowest_bic = np.infty
        best_model = None

        for n_comps in range(1, max_comps + 1):
            # Fixing random_state will break the generate method (samples single data points).
            gmm = GaussianMixture(n_components=n_comps, covariance_type=self.cov_type)
            gmm.fit(data)
            current_bic = gmm.bic(data)
            if current_bic < lowest_bic:
                lowest_bic = current_bic
                best_model = gmm

        # Fix - single component GMMs can have incorrect weights.
        if len(best_model.weights_) == 1:
            best_model.weights_[0] = min(best_model.weights_[0], 1.0)

        return best_model
    
    def _fit_subgenerators(self, splitter, input, all_data):
        all_estimators = []
        estimators = splitter.estimators_ if hasattr(splitter, 'estimators_') else [splitter]

        all_missed = []
        for estimator in estimators:
            sub_pop_ids = estimator.apply(input)
            probs = []
            models = []
            missed = 0

            for id in set(sub_pop_ids):
                sub_pop = all_data[sub_pop_ids == id]
                # 2 data-points is a bare minimum to fit a GMM.
                if sub_pop.shape[0] > 1:
                    max_comps_sub = min(sub_pop.shape[0], self.n_components)
                    g = self._find_best_gmm(sub_pop, max_comps_sub)
                    probs.append(np.sum(sub_pop_ids == id) / len(sub_pop_ids))
                    models.append(g)
                else:
                    missed += 1

            if len(probs) > 0:
                n_probs_arr = 1.0 - np.array(probs)
                inv_probs = list(n_probs_arr / n_probs_arr.sum())
                all_estimators.append([(probs, inv_probs), models])

            if missed > 0:    
                all_missed.append(missed)
        
        if len(all_missed) > 0 and self.verbose > 1:
            self.logger.info(f'Missed data points: {all_missed}')

        return all_estimators
    
    def _bound_cat_vars(self, arr):
        arr[:, self.cat_vars] = arr[:, self.cat_vars] > 0.5
    
    def _generate(self, estimators, n_samples):
        def process():
            (probs, inv_probs), models = random.choice(estimators)

            if self.gen_mode == 'uni':
                m = np.random.choice(models)
            elif self.gen_mode == 'inv':
                m = np.random.choice(models, p=inv_probs)
            else: # 'obs'
                m = np.random.choice(models, p=probs)

            return (m.sample()[0])

        # n_jobs=1 is faster than -1
        results = Parallel(n_jobs=1)(delayed(process)() for _ in range(n_samples))

        return np.array(np.squeeze(results))

class BalancedTEstimator():
    def __init__(self, estimator='auto', n_estimators=1, max_depth=None, min_samples_leaf=1, split_mode='x-x', gen_mode='uni', gen_samples=-1, gen_noise=0.01, incl_real=False, n_components=1, cov_type='full', cat_vars=None, random_state=None, n_jobs=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.split_mode = split_mode
        self.gen_mode = gen_mode
        self.gen_samples = gen_samples
        self.gen_noise = gen_noise
        self.incl_real = incl_real
        self.n_components = n_components
        self.cov_type = cov_type
        self.cat_vars = cat_vars
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._init_generators()

    def _init_generators(self):
        self.generator_0 = BalancingGenerator(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, split_mode=self.split_mode, gen_mode=self.gen_mode, n_components=self.n_components, cov_type=self.cov_type, cat_vars=self.cat_vars, random_state=self.random_state, n_jobs=self.n_jobs, verbose=self.verbose)
        self.generator_1 = BalancingGenerator(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, split_mode=self.split_mode, gen_mode=self.gen_mode, n_components=self.n_components, cov_type=self.cov_type, cat_vars=self.cat_vars, random_state=self.random_state, n_jobs=self.n_jobs, verbose=self.verbose)
        self.logger = logging.getLogger('models.BalancedTEstimator')
    
    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        self._init_generators()
        return self
    
    def get_params(self, deep=True):
        return {'estimator': self.estimator, 'gen_samples': self.gen_samples, 'gen_noise': self.gen_noise, 'incl_real': self.incl_real, 'split_mode': self.split_mode, 'gen_mode': self.gen_mode, 'n_estimators': self.n_estimators, 'n_components': self.n_components, 'cov_type': self.cov_type, 'min_samples_leaf': self.min_samples_leaf, 'max_depth': self.max_depth, 'cat_vars': self.cat_vars, 'random_state': self.random_state, 'n_jobs': self.n_jobs, 'verbose': self.verbose}

    def fit(self, X, y):
        if self.estimator == 'auto':
            self.estimator = ExtraTreesRegressor(n_jobs=self.n_jobs, random_state=self.random_state)
        
        y_2d = y.reshape(-1, 1)
        t_1d = X[:, -1]
        X_only = X[:, :-1]

        self.generator_0.fit(X_only[t_1d == 0], None, y_2d[t_1d == 0])
        self.generator_1.fit(X_only[t_1d == 1], None, y_2d[t_1d == 1])

        if self.gen_samples < 0:
            # Find optimal sample size
            if self.split_mode == 'x-x':
                X_in = np.concatenate([X_only, y_2d], axis=1)
            else:
                X_in = X_only
            size0, n_bucks0 = self.generator_0.bucket_size(X_in[t_1d == 0])
            size1, n_bucks1 = self.generator_1.bucket_size(X_in[t_1d == 1])
            n0 = n1 = (min(size0, size1) * (n_bucks0 + n_bucks1)) // 2
        elif self.gen_samples > 1:
            n0 = n1 = self.gen_samples // 2
        else: # <0, 1> - fraction
            n0 = n1 = int((self.gen_samples * X.shape[0]) // 2)

        if self.verbose > 1:
            percentage = ((n0 + n1) / X.shape[0]) * 100.0
            self.logger.info(f'Generated {n0 + n1} samples ({percentage:.2f}%)')

        gen_data_0 = self.generator_0.sample(n0, self.gen_noise)
        gen_data_1 = self.generator_1.sample(n1, self.gen_noise)
        
        gen_data_t0 = np.concatenate([gen_data_0[:, :-1], np.zeros((gen_data_0.shape[0], 1)), gen_data_0[:, -1:]], axis=1)
        gen_data_t1 = np.concatenate([gen_data_1[:, :-1], np.ones((gen_data_1.shape[0], 1)), gen_data_1[:, -1:]], axis=1)

        gen_data = np.vstack((gen_data_t0, gen_data_t1))

        if self.incl_real:
            gen_data = np.vstack((gen_data, np.concatenate([X, y_2d], axis=1)))

        self.X_gen = gen_data[:, :-1]
        self.y_gen = gen_data[:, -1]
        self.estimator.fit(self.X_gen, self.y_gen)
    
        return self

    def predict(self, X):
        return self.estimator.predict(X)
    
    def score(self, X, y):
        return self.estimator.score(X, y)
    
    def get_data(self):
        return self.X_gen, self.y_gen

class BalancedTEstimatorMeta():
    def __init__(self, estimator='auto', n_estimators=1, max_depth=None, min_samples_leaf=1, split_mode='x-x', gen_mode='uni', gen_samples=-1, gen_noise=0.01, incl_real=False, n_components=1, cov_type='full', cat_vars=None, random_state=None, n_jobs=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.split_mode = split_mode
        self.gen_mode = gen_mode
        self.gen_samples = gen_samples
        self.gen_noise = gen_noise
        self.incl_real = incl_real
        self.n_components = n_components
        self.cov_type = cov_type
        self.cat_vars = cat_vars
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._init_generators()

    def _init_generators(self):
        self.generator_0 = BalancingGenerator(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, split_mode=self.split_mode, gen_mode=self.gen_mode, n_components=self.n_components, cov_type=self.cov_type, cat_vars=self.cat_vars, random_state=self.random_state, n_jobs=self.n_jobs, verbose=self.verbose)
        self.generator_1 = BalancingGenerator(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, split_mode=self.split_mode, gen_mode=self.gen_mode, n_components=self.n_components, cov_type=self.cov_type, cat_vars=self.cat_vars, random_state=self.random_state, n_jobs=self.n_jobs, verbose=self.verbose)
        self.logger = logging.getLogger('models.BalancedTEstimator')
    
    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        self._init_generators()
        return self
    
    def get_params(self, deep=True):
        return {'estimator': self.estimator, 'gen_samples': self.gen_samples, 'gen_noise': self.gen_noise, 'incl_real': self.incl_real, 'split_mode': self.split_mode, 'gen_mode': self.gen_mode, 'n_estimators': self.n_estimators, 'n_components': self.n_components, 'cov_type': self.cov_type, 'min_samples_leaf': self.min_samples_leaf, 'max_depth': self.max_depth, 'cat_vars': self.cat_vars, 'random_state': self.random_state, 'n_jobs': self.n_jobs, 'verbose': self.verbose}

    def fit(self, X, y):
        if self.estimator == 'auto':
            self.estimator = ExtraTreesRegressor(n_jobs=self.n_jobs, random_state=self.random_state)
        
        y_2d = y.reshape(-1, 1)
        t_1d = X[:, -1]
        X_only = X[:, :-1]

        self.generator_0.fit(X_only[t_1d == 0], None, y_2d[t_1d == 0])
        self.generator_1.fit(X_only[t_1d == 1], None, y_2d[t_1d == 1])

        if self.gen_samples < 0:
            # Find optimal sample size
            if self.split_mode == 'x-x':
                X_in = np.concatenate([X_only, y_2d], axis=1)
            else:
                X_in = X_only
            size0, n_bucks0 = self.generator_0.bucket_size(X_in[t_1d == 0])
            size1, n_bucks1 = self.generator_1.bucket_size(X_in[t_1d == 1])
            n0 = n1 = (min(size0, size1) * (n_bucks0 + n_bucks1)) // 2
        elif self.gen_samples > 1:
            n0 = n1 = self.gen_samples // 2
        else: # <0, 1> - fraction
            n0 = n1 = int((self.gen_samples * X.shape[0]) // 2)

        if self.verbose > 1:
            percentage = ((n0 + n1) / X.shape[0]) * 100.0
            self.logger.info(f'Generated {n0 + n1} samples ({percentage:.2f}%)')

        gen_data_0 = self.generator_0.sample(n0, self.gen_noise)
        gen_data_1 = self.generator_1.sample(n1, self.gen_noise)
        
        gen_data_t0 = np.concatenate([gen_data_0[:, :-1], np.zeros((gen_data_0.shape[0], 1)), gen_data_0[:, -1:]], axis=1)
        gen_data_t1 = np.concatenate([gen_data_1[:, :-1], np.ones((gen_data_1.shape[0], 1)), gen_data_1[:, -1:]], axis=1)

        gen_data = np.vstack((gen_data_t0, gen_data_t1))

        if self.incl_real:
            gen_data = np.vstack((gen_data, np.concatenate([X, y_2d], axis=1)))

        self.X_gen = gen_data[:, :-2]
        self.t_gen = gen_data[:, -2]
        self.y_gen = gen_data[:, -1]
        self.estimator.fit(Y=self.y_gen, T=self.t_gen, X=self.X_gen)
    
        return self

    def effect(self, X, T0, T1):
        return self.estimator.effect(X=X, T0=T0, T1=T1)
    
    def get_data(self):
        return self.X_gen, self.t_gen, self.y_gen

class BalancedEstimatorSearch():
    def __init__(self, search_estimator, final_estimator, param_grid, n_estimators=1, max_depth=None, min_samples_leaf=1, split_mode='x-x', gen_mode='uni', gen_samples=-1, gen_noise=0.01, incl_real=False, n_components=1, cov_type='full', cat_vars=None, cv=5, random_state=None, n_jobs=None, verbose=0):
        self.search_estimator = search_estimator
        self.final_estimator = final_estimator
        self.param_grid = param_grid
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.split_mode = split_mode
        self.gen_mode = gen_mode
        self.gen_samples = gen_samples
        self.gen_noise = gen_noise
        self.incl_real = incl_real
        self.n_components = n_components
        self.cov_type = cov_type
        self.cat_vars = cat_vars
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def fit(self, X, y):
        balanced_generator = BalancedTEstimator(estimator=self.search_estimator, n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, split_mode=self.split_mode, gen_mode=self.gen_mode, gen_samples=self.gen_samples, gen_noise=self.gen_noise, incl_real=self.incl_real, n_components=self.n_components, cov_type=self.cov_type, cat_vars=self.cat_vars, random_state=self.random_state, n_jobs=1, verbose=self.verbose)
        grid_search = GridSearchCV(balanced_generator, param_grid=self.param_grid, scoring='neg_mean_squared_error', n_jobs=self.n_jobs, cv=self.cv)
        grid_search.fit(X, y)
        X_new, y_new = grid_search.best_estimator_.get_data()
        self.final_estimator.fit(X_new, y_new)
        return self

    def predict(self, X):
        return self.final_estimator.predict(X)