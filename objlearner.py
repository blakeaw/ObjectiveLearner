"""Implements decorators for objective functions used in calibrations.

This module defines the three decorator classes:
    * ObjectiveCounter - counts the number of objective function calls.
    * ObjectiveSaver - saves the input-output pairs fo the objective function calls.
    * ObjectiveLearner - provides machine learning (linear regression) and sensitivity analysis.
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR
#import SALib
from SALib.analyze import sobol,morris,delta,fast,rbd_fast
from SALib.sample import saltelli,morris,latin,fast_sampler

class ObjectiveCounter(object):
    """Counts the number of objective function calls.
    This class is to used a decorator of the objective function in calibrations and
    is used to count the number of function calls.

    Attributes:
        count (int): The number of objective function calls.
    """
    count = 0
    _objective_function = None
    def __init__(self, objective_function):
        """Initialize the ObjectiveCounter.

        Args:
            objective_function (function): The objective function.
        """
        self._objective_function = objective_function

    def __call__(self, theta):
        """ObjectiveCounter call that wraps of the objective function.

        Args:
            theta (numpy.ndarray): The parameter vector to be evaluated by the
                objective function.
        """
        objective = self._objective_function(theta)
        self.count += 1
        return objective

class ObjectiveSaver(ObjectiveCounter):
    """Saves the input-output pairs fo the objective function calls.
    This class is to used a decorator of the objective function in calibrations and
    is used to save the input and outputs of calls to the objective function.

    Attributes:
        count (int): The number of objective function calls.
        objective_theta (pandas.DataFrame): The objective function values and the input
            parameter vectors.
    """
    _objective_theta = None
    def __init__(self, objective_function):
        """Initialize the ObjectiveSaver.

        Args:
            objective_function (function): The objective function.
        """
        self._objective_function = objective_function
        self._objective_theta = list()

    def __call__(self, theta):
        """ObjectiveSaver call that wraps of the objective function.

        Args:
            theta (numpy.ndarray): The parameter vector to be evaluated by the
                objective function.
        """
        objective = self._objective_function(theta)
        dpd = dict({'objective': objective, })
        for k,val in enumerate(theta):
            dpd[k] = val
        self._objective_theta.append(dpd)
        self.count += 1
        return objective

    def write_csv(self, prefix='objective_data'):
        """Write the objective function and parameter vector DataFrame values to a csv file.
        """
        objective_theta = self.objective_theta
        objective_theta.to_csv(prefix+".csv", compression='zip')
        return

    def write_npy(self, prefix='objective_data'):
        """Write the objective function and parameter vector DataFrame values to a NumPy npy file.
        """
        objective_theta = self.objective_theta
        np.save(prefix, objective_theta.values, allow_pickle=False)
        return

    @property
    def objective_theta(self):
        """pandas.DataFrame: objective function and parameter vector values.
        """
        return pd.DataFrame(self._objective_theta)


class ObjectiveLearner(ObjectiveSaver):
    """Provides machine learning (linear regression) and sensitivity analysis.
    This class is used as a decorator of the objective function in calibrations and
    is used to save the input and outputs of calls to the objective function while
    providing functions to run machine learning (linear regression) and
    sensitivity analyses of the objective function.

    Attributes:
        count (int): The number of objective function calls.
        objective_theta (pandas.DataFrame): The objective function values and the input
            parameter vectors.
    """
    #def __init__(self, objective_function):
    #    super(ObjectiveSaver, self).__init__(objective_function)
    #def __call__(self, theta):
    #    return super(ObjectiveSaver, self).__call__(theta)
    def __init__(self, objective_function):
        """Initialize the ObjectiveLearner.

        Args:
            objective_function (function): The objective function.
        """
        self._objective_function = objective_function
        self._objective_theta = list()

    def __call__(self, theta):
        """ObjectiveLearner call that wraps of the objective function.

        Args:
            theta (numpy.ndarray): The parameter vector to be evaluated by the
                objective function.
        """
        objective = self._objective_function(theta)
        dpd = dict({'objective': objective, })
        for k,val in enumerate(theta):
            dpd[k] = val
        self._objective_theta.append(dpd)
        self.count += 1
        return objective

    def best_data(self, n_points=100, cost=True):
        """Gets the set of highest objective and parameter data.
        """
        # Load the dataset -- panda DataFrame
        objective_theta = self.objective_theta
        # Sort
        objective_theta.sort_values(by=['objective'], inplace=True)
        # Split into X and Y
        # X is set of theta vectors
        idx_X = objective_theta.columns[1:]
        X = objective_theta[idx_X]
        # y is the objective
        y = objective_theta['objective'].to_numpy()
        if cost:
            X_best = X[:n_points]
            y_best = y[:n_points]
        else:
            X_best = X[-n_points:]
            y_best = y[-n_points:]
        return X_best.to_numpy(), y_best

    def split_data(self):
        """Splits the objective and paramter data into training and test sets.
        """
        # Load the dataset -- panda DataFrame
        objective_theta = self.objective_theta
        # Shuffle -- https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        objective_theta = objective_theta.sample(frac=1).reset_index(drop=True)
        # Split into X and Y
        # X is set of theta vectors
        idx_X = objective_theta.columns[1:]
        X = objective_theta[idx_X]
        # y is the objective
        y = objective_theta['objective'].to_numpy()
        n_y = len(y)
        # Split into training/testing sets
        tenp = int(n_y*0.10)
        X_train = X[:-tenp]
        X_test = X[-tenp:]
        y_train = y[:-tenp]
        y_test = y[-tenp:]
        return X_train, X_test, y_train, y_test

    def _linear_regression(self, model, *args, **kwargs):
        """Runs linear regression of the objective function against the parameters.
        This function estimates the coefficients and the explained variance
        score for a linear least squares fit of the objective function vs. the
        input parameters using the given regression model class from scikit-learn.

        Returns:
            tuple(numpy.ndarray, float): coefficients, explained_variance_score.
        """
        X_train, X_test, y_train, y_test = self.split_data()
        # Initialize the model
        regr = model(*args, **kwargs)
        # Train the model
        regr.fit(X_train, y_train)
        # Score the predictions -- score function is the explained variance score.
        ev_score = regr.score(X_test, y_test)
        return regr.coef_, ev_score

    def least_squares(self, *args, **kwargs):
        """Least squares linear regression of the objective function against the parameters.
        This function estimates the coefficients and the explained variance
        score for a linear least squares fit of the objective function vs. the
        input parameters using the LinearRegression class from scikit-learn.

        Returns:
            tuple(numpy.ndarray, float): coefficients, explained_variance_score.
        """
        return self._linear_regression(linear_model.LinearRegression, *args, **kwargs)

    def ridge(self, *args, **kwargs):
        """Ridge regression of the objective function against the parameters.
        This function estimates the coefficients and the explained variance
        score for a Ridge regression fit of the objective function vs. the
        input parameters using the Ridge class from scikit-learn.

        Returns:
            tuple(numpy.ndarray, float): coefficients, explained_variance_score.
        """
        return self._linear_regression(linear_model.Ridge, *args, **kwargs)

    def lasso(self, *args, **kwargs):
        """Lasso regression of the objective function against the parameters.
        This function estimates the coefficients and the explained variance
        score for a Lasso regression fit of the objective function vs. the
        input parameters using the Lasso class from scikit-learn.

        Returns:
            tuple(numpy.ndarray, float): coefficients, explained_variance_score.
        """
        return self._linear_regression(linear_model.Lasso, *args, **kwargs)

    def linear_svr(self, *args, **kwargs):
        """Linear Support Vector Regression (SVR) of the objective function against the parameters.
        This function estimates the coefficients and the explained variance
        score for a SVR fit of the objective function vs. the
        input parameters using the SVR class from scikit-learn.

        Returns:
            tuple(numpy.ndarray, float): coefficients, explained_variance_score.
        """
        if 'gamma' not in kwargs.keys():
            kwargs['gamma'] = 'auto'
        kwargs['kernel'] = 'linear'
        return self._linear_regression(SVR, *args, **kwargs)

    def harmonic_local_optimum_sensitivity(self, cost=True):
        """Sensitivity measure around the optimum of the objective function from regression.
        This function estimates the coefficients and the explained variance
        score for a linear least squares fit of the objective function vs. the
        input parameters using the LinearRegression class from scikit-learn.

        Returns:
            tuple(numpy.ndarray, float): coefficients, explained_variance_score.
        """
        X_best, y_best = self.best_data(n_points=int(0.10*self.count), cost=cost)
        if cost:
            X_diff = X_best - X_best[0]
            #print(X_best[-1])
            y_diff = y_best - y_best.min()
        else:
            X_diff = X_best - X_best[-1]
            #print(X_best[-1])
            y_diff = y_best - y_best.max()
        #print(y_diff)
        X_sq = X_diff**2
        X_sq = X_sq/X_sq.max()
        #print(X_sq)
        # Initialize the model -- Let's use Lasso with large alpha
        regr = linear_model.Lasso(alpha=0.25, fit_intercept=False)
        #regr = linear_model.LinearRegression()
        # Train the model
        regr.fit(X_sq, y_diff)
        coef = np.abs(regr.coef_)
        coef/coef.sum()
        fmask = coef < 0.001
        coef[fmask] = 0.
        return coef/coef.sum()

    def _sensitivity_prep(self):
        # Load the dataset -- pandas DataFrame
        objective_theta = self.objective_theta
        #objective_theta.sort_values(by=['objective'], inplace=True)
        objective_theta.sample(frac=1).reset_index(drop=True)
        # Split into X and Y
        # X is set of theta vectors
        idx_X = objective_theta.columns[1:]
        X = objective_theta[idx_X].to_numpy()
        # y is the objective
        y = objective_theta['objective'].to_numpy()
        n_X = len(X[0])
        names = list()
        bounds = list()
        for i in range(n_X):
            X_i = X[:,i]
            X_i_max = X_i.max()
            X_i_min = X_i.min()
            names.append(i)
            bound = list([X_i_min, X_i_max])
            bounds.append(bound)

        problem = dict({'num_vars': n_X, 'names': names, 'bounds': bounds})
        return X, y, problem

    def _closest_points(self, problem, X, y, samples):
        n_samples = len(samples)
        X_s = np.zeros((n_samples,problem['num_vars']))
        y_s = np.zeros(n_samples)
        for i,theta in enumerate(samples):
            dists = np.linalg.norm(X - theta, axis=1)
            idx = np.argmin(dists)
            #print(theta, X[idx], dists[idx])
            X_s[i] = X[idx]
            y_s[i] = y[idx]
        return X_s, y_s

    def sobol(self):
        """Sobol sensitivity of the objective function.
        This function estimates the Sobol sensitivity indicies of the
        objective function with changes in the parameters using SALib:
        https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis

        Returns:
            dict: sensitivity indices of parameters; dict has keys 'S1',
                'S1_conf', 'ST', and 'ST_conf'
        """
        X, y, problem = self._sensitivity_prep()
        n_sample = 2000
        param_values = saltelli.sample(problem, n_sample)
        X_s, y_s = self._closest_points(problem, X, y, param_values)
        Si = sobol.analyze(problem, y_s)
        return Si

    def morris(self):
        """Morris Method sensitivity of the objective function.
        This function estimates the sensitivity with the Morris Method of the
        objective function with changes in the parameters using SALib:
        https://salib.readthedocs.io/en/latest/api.html#method-of-morris

        Returns:
            dict: sensitivity values of parameters; dict has keys 'mu',
                'mu_star', 'sigma', and 'mu_star_conf'
        """
        X, y, problem = self._sensitivity_prep()
        n_sample = 2000
        param_values = morris.sample(problem, n_sample)
        X_s, y_s = self._closest_points(problem, X, y, param_values)
        Si = morris.analyze(problem, X_s, y_s)
        return Si

    def delta(self):
        """Morris Method sensitivity of the objective function.
        This function estimates the sensitivity with the Morris Method of the
        objective function with changes in the parameters using SALib:
        https://salib.readthedocs.io/en/latest/api.html#delta-moment-independent-measure

        Returns:
            dict: sensitivity values of parameters; dict has keys 'delta',
                'delta_conf', 'S1', and 'S1_conf'
        """
        X, y, problem = self._sensitivity_prep()
        n_sample = 2000
        param_values = latin.sample(problem, n_sample)
        X_s, y_s = self._closest_points(problem, X, y, param_values)
        Si = delta.analyze(problem, X_s, y_s)
        return Si

    def fast(self):
        """FAST sensitivity analysis of the objective function.
        This function estimates the sensitivity with the FAST method of the
        objective function with changes in the parameters using SALib:
        https://salib.readthedocs.io/en/latest/api.html#fast-fourier-amplitude-sensitivity-test

        Returns:
            dict: sensitivity values of parameters; dict has keys 'S1' and 'ST'
        """
        X, y, problem = self._sensitivity_prep()
        n_sample = 2000
        param_values = fast_sampler.sample(problem, n_sample)
        X_s, y_s = self._closest_points(problem, X, y, param_values)
        Si = fast.analyze(problem, y_s)
        return Si

    def rbd_fast(self):
        """RBD-FAST sensitivity analysis of the objective function.
        This function estimates the sensitivity with the RBD-FAST method of the
        objective function with changes in the parameters using SALib:
        https://salib.readthedocs.io/en/latest/api.html#rbd-fast-random-balance-designs-fourier-amplitude-sensitivity-test

        Returns:
            dict: sensitivity values of parameters; dict has keys 'S1'
        """
        X, y, problem = self._sensitivity_prep()
        n_sample = 2000
        param_values = latin.sample(problem, n_sample)
        X_s, y_s = self._closest_points(problem, X, y, param_values)
        Si = rbd_fast.analyze(problem, X_s, y_s)
        return Si
