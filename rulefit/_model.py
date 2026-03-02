"""RuleFit estimator implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from ._rules import RuleEnsemble
from ._scaling import FriedScale, Winsorizer


class RuleFit(BaseEstimator, TransformerMixin):
    """Linear model of extracted decision rules and optional raw features."""

    def __init__(
        self,
        tree_size=4,
        sample_fract="default",
        max_rules=2000,
        memory_par=0.01,
        tree_generator=None,
        default_tree_generator="random_forest",
        rfmode="regress",
        lin_trim_quantile=0.025,
        lin_standardise=True,
        exp_rand_tree_size=True,
        model_type="rl",
        Cs=None,
        cv=3,
        tol=0.0001,
        max_iter=None,
        n_jobs=None,
        random_state=None,
    ):
        self.tree_generator = tree_generator
        self.default_tree_generator = default_tree_generator
        self.rfmode = rfmode
        self.lin_trim_quantile = lin_trim_quantile
        self.lin_standardise = lin_standardise
        self.winsorizer = Winsorizer(trim_quantile=lin_trim_quantile)
        self.friedscale = FriedScale(self.winsorizer)
        self.stddev = None
        self.mean = None
        self.exp_rand_tree_size = exp_rand_tree_size
        self.max_rules = max_rules
        self.sample_fract = sample_fract
        self.memory_par = memory_par
        self.tree_size = tree_size
        self.random_state = random_state
        self.model_type = model_type
        self.cv = cv
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.Cs = Cs

    def _default_sample_fract(self, n_samples: int) -> float:
        return min(0.5, (100.0 + 6.0 * np.sqrt(n_samples)) / n_samples)

    def _init_default_tree_generator(self, n_samples: int):
        if self.default_tree_generator not in {"random_forest", "gradient_boosting"}:
            raise ValueError(
                "default_tree_generator must be either 'random_forest' or "
                "'gradient_boosting'"
            )

        n_estimators = int(np.ceil(self.max_rules / self.tree_size))
        sample_fract = (
            self._default_sample_fract(n_samples)
            if self.sample_fract == "default"
            else self.sample_fract
        )
        self.sample_fract_ = sample_fract

        if self.default_tree_generator == "random_forest":
            common = dict(
                n_estimators=n_estimators,
                max_leaf_nodes=self.tree_size,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            if self.rfmode == "regress":
                self.tree_generator = RandomForestRegressor(**common)
            else:
                self.tree_generator = RandomForestClassifier(**common)
            return

        common = dict(
            n_estimators=n_estimators,
            max_leaf_nodes=self.tree_size,
            learning_rate=self.memory_par,
            subsample=sample_fract,
            random_state=self.random_state,
            max_depth=100,
        )
        if self.rfmode == "regress":
            self.tree_generator = GradientBoostingRegressor(**common)
        else:
            self.tree_generator = GradientBoostingClassifier(**common)

    def _validate_tree_generator_type(self) -> None:
        if self.rfmode == "regress":
            valid = (GradientBoostingRegressor, RandomForestRegressor)
            if not isinstance(self.tree_generator, valid):
                raise ValueError(
                    "RuleFit only works with RandomForestRegressor or GradientBoostingRegressor"
                )
        else:
            valid = (GradientBoostingClassifier, RandomForestClassifier)
            if not isinstance(self.tree_generator, valid):
                raise ValueError(
                    "RuleFit only works with RandomForestClassifier or GradientBoostingClassifier"
                )

    def _fit_tree_generator(self, x: np.ndarray, y: np.ndarray) -> None:
        supports_randomized_growth = isinstance(self.tree_generator, GradientBoostingRegressor)
        if (not self.exp_rand_tree_size) or (not supports_randomized_growth):
            self.tree_generator.fit(x, y)
            return

        rng = np.random.RandomState(self.random_state)
        sizes = rng.exponential(
            scale=self.tree_size - 2,
            size=int(np.ceil(self.max_rules * 2 / self.tree_size)),
        )
        tree_sizes = np.asarray(2 + np.floor(sizes), dtype=int)
        i = int(len(tree_sizes) / 4)
        while np.sum(tree_sizes[:i]) < self.max_rules and i < len(tree_sizes):
            i += 1
        tree_sizes = tree_sizes[:i]

        self.tree_generator.set_params(warm_start=True)
        curr_estimators = 0

        random_state_add = self.random_state if self.random_state is not None else 0
        for idx, size in enumerate(tree_sizes):
            self.tree_generator.set_params(n_estimators=curr_estimators + 1)
            if "max_leaf_nodes" in self.tree_generator.get_params(deep=False):
                self.tree_generator.set_params(max_leaf_nodes=int(size))
            self.tree_generator.set_params(random_state=idx + random_state_add)
            self.tree_generator.fit(np.copy(x, order="C"), np.copy(y, order="C"))
            curr_estimators += 1

        self.tree_generator.set_params(warm_start=False)

    def _extract_rule_ensemble(self, feature_names) -> None:
        tree_list = self.tree_generator.estimators_
        if isinstance(self.tree_generator, (RandomForestRegressor, RandomForestClassifier)):
            tree_list = [[tree] for tree in self.tree_generator.estimators_]
        self.rule_ensemble = RuleEnsemble(tree_list=tree_list, feature_names=feature_names)

    def _build_feature_matrix(self, x: np.ndarray, include_rules: bool = True) -> np.ndarray:
        blocks = []

        if "l" in self.model_type:
            if self.lin_standardise:
                blocks.append(self.friedscale.scale(x))
            else:
                blocks.append(x)

        if "r" in self.model_type and include_rules:
            n_rules = len(self.rule_ensemble.rules)
            rule_coefs = self.coef_[-n_rules:] if hasattr(self, "coef_") else None
            x_rules = self.rule_ensemble.transform(x, coefs=rule_coefs)
            if x_rules.shape[1] > 0:
                blocks.append(x_rules)

        if not blocks:
            return np.zeros((x.shape[0], 0))

        return np.concatenate(blocks, axis=1)

    def fit(self, X, y=None, feature_names=None):
        """Fit RuleFit model."""
        x = np.asarray(X)
        y_arr = np.asarray(y)
        n_samples, n_features = x.shape

        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            self.feature_names = list(feature_names)

        if "r" in self.model_type:
            if self.tree_generator is None:
                self._init_default_tree_generator(n_samples)
            else:
                for attr in ("estimators_",):
                    if hasattr(self.tree_generator, attr):
                        delattr(self.tree_generator, attr)

            self._validate_tree_generator_type()
            self._fit_tree_generator(x, y_arr)
            self._extract_rule_ensemble(feature_names=self.feature_names)

        if "l" in self.model_type:
            self.winsorizer.train(x)
            winsorized = self.winsorizer.trim(x)
            self.stddev = np.std(winsorized, axis=0)
            self.mean = np.mean(winsorized, axis=0)
            if self.lin_standardise:
                self.friedscale.train(x)

        x_concat = self._build_feature_matrix(x, include_rules=True)

        if self.rfmode == "regress":
            max_iter = 1000 if self.max_iter is None else self.max_iter
            if self.Cs is None:
                alphas = 100
            elif hasattr(self.Cs, "__len__"):
                alphas = 1.0 / np.asarray(self.Cs)
            else:
                alphas = int(self.Cs)
            self.lscv = LassoCV(
                alphas=alphas,
                cv=self.cv,
                max_iter=max_iter,
                tol=self.tol,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            self.lscv.fit(x_concat, y_arr)
            self.coef_ = self.lscv.coef_
            self.intercept_ = self.lscv.intercept_
        else:
            max_iter = 1000 if self.max_iter is None else self.max_iter
            Cs = 10 if self.Cs is None else self.Cs
            self.lscv = LogisticRegressionCV(
                Cs=Cs,
                cv=self.cv,
                l1_ratios=(1,),
                max_iter=max_iter,
                tol=self.tol,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                solver="liblinear",
                use_legacy_attributes=False,
            )
            self.lscv.fit(x_concat, y_arr)
            self.coef_ = self.lscv.coef_[0]
            self.intercept_ = self.lscv.intercept_[0]

        return self

    def predict(self, X):
        x_concat = self._build_feature_matrix(np.asarray(X), include_rules=True)
        return self.lscv.predict(x_concat)

    def predict_proba(self, X):
        if "predict_proba" not in dir(self.lscv):
            raise ValueError(
                "Probability prediction using predict_proba not available for model type "
                f"{self.lscv}"
            )
        x_concat = self._build_feature_matrix(np.asarray(X), include_rules=True)
        return self.lscv.predict_proba(x_concat)

    def transform(self, X=None, y=None):
        return self.rule_ensemble.transform(np.asarray(X))

    @staticmethod
    def _format_rule(rule, round_digits: int | None) -> str:
        parts = []
        for cond in rule.conditions:
            feature = cond.feature_name if cond.feature_name is not None else cond.feature_index
            threshold = round(cond.threshold, round_digits) if round_digits is not None else cond.threshold
            parts.append(f"{feature} {cond.operator} {threshold}")
        return " & ".join(parts)

    @staticmethod
    def _rule_base_name(rule) -> str:
        """Sorted unique feature names from a rule's conditions, joined by '_'."""
        seen: dict[str, None] = {}
        for cond in rule.conditions:
            name = str(cond.feature_name if cond.feature_name is not None else cond.feature_index)
            seen[name] = None
        return "_".join(sorted(seen))

    def get_rules(self, exclude_zero_coef=False, round_digits=2) -> pd.DataFrame:
        """Return fitted rules and linear terms as a DataFrame.

        Parameters
        ----------
        exclude_zero_coef : bool, default False
            If True, only return rules/features with non-zero coefficients.
        round_digits : int or None, default 2
            Round rule thresholds to this many decimal places for readability.
            Pass None to keep the full floating-point precision.

        Returns
        -------
        pd.DataFrame with columns:
            rule         : string representation of the rule or feature name
            type         : "rule" or "linear"
            feature_name : underscore-joined feature names used in the rule;
                           enumerated (e.g. age_income_0) when names collide
            coef         : coefficient from the linear model
            support      : fraction of training samples satisfying the rule
                           (1.0 for linear features)
            importance   : abs(coef) * support
        """
        records = []
        coef_idx = 0

        if "l" in self.model_type:
            for name in self.feature_names:
                coef = self.coef_[coef_idx]
                records.append({
                    "rule": name,
                    "type": "linear",
                    "feature_name": name,
                    "coef": coef,
                    "support": 1.0,
                    "importance": abs(coef),
                })
                coef_idx += 1

        if "r" in self.model_type:
            for rule in self.rule_ensemble.rules:
                coef = self.coef_[coef_idx]
                records.append({
                    "rule": self._format_rule(rule, round_digits),
                    "type": "rule",
                    "feature_name": self._rule_base_name(rule),
                    "coef": coef,
                    "support": rule.support,
                    "importance": abs(coef) * rule.support,
                })
                coef_idx += 1

        # Enumerate colliding feature_name values among rules only.
        # Linear terms are unique by construction and keep their plain name.
        # e.g. two rules both using age+income → age_income_0, age_income_1
        rule_name_counts: dict[str, int] = {}
        for r in records:
            if r["type"] == "rule":
                rule_name_counts[r["feature_name"]] = rule_name_counts.get(r["feature_name"], 0) + 1

        name_seen: dict[str, int] = {}
        for r in records:
            if r["type"] == "rule" and rule_name_counts[r["feature_name"]] > 1:
                base = r["feature_name"]
                idx = name_seen.get(base, 0)
                r["feature_name"] = f"{base}_{idx}"
                name_seen[base] = idx + 1

        cols = ["rule", "type", "feature_name", "coef", "support", "importance"]
        rules_df = pd.DataFrame(records, columns=cols)

        if exclude_zero_coef:
            rules_df = rules_df[rules_df.coef != 0].reset_index(drop=True)

        return rules_df


class RuleFitRegressor(RuleFit, RegressorMixin, TransformerMixin):
    def __init__(
        self,
        tree_size=4,
        sample_fract="default",
        max_rules=2000,
        memory_par=0.01,
        tree_generator=None,
        default_tree_generator="random_forest",
        rfmode="regress",
        lin_trim_quantile=0.025,
        lin_standardise=True,
        exp_rand_tree_size=True,
        model_type="rl",
        Cs=None,
        cv=3,
        tol=0.0001,
        max_iter=1000,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            tree_size=tree_size,
            sample_fract=sample_fract,
            max_rules=max_rules,
            memory_par=memory_par,
            tree_generator=tree_generator,
            default_tree_generator=default_tree_generator,
            rfmode=rfmode,
            lin_trim_quantile=lin_trim_quantile,
            lin_standardise=lin_standardise,
            exp_rand_tree_size=exp_rand_tree_size,
            model_type=model_type,
            Cs=Cs,
            cv=cv,
            tol=tol,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class RuleFitClassifier(RuleFit, ClassifierMixin, TransformerMixin):
    def __init__(
        self,
        tree_size=4,
        sample_fract="default",
        max_rules=2000,
        memory_par=0.01,
        tree_generator=None,
        default_tree_generator="random_forest",
        rfmode="classify",
        lin_trim_quantile=0.025,
        lin_standardise=True,
        exp_rand_tree_size=True,
        model_type="rl",
        Cs=None,
        cv=3,
        tol=0.0001,
        max_iter=1000,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            tree_size=tree_size,
            sample_fract=sample_fract,
            max_rules=max_rules,
            memory_par=memory_par,
            tree_generator=tree_generator,
            default_tree_generator=default_tree_generator,
            rfmode=rfmode,
            lin_trim_quantile=lin_trim_quantile,
            lin_standardise=lin_standardise,
            exp_rand_tree_size=exp_rand_tree_size,
            model_type=model_type,
            Cs=Cs,
            cv=cv,
            tol=tol,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=random_state,
        )
