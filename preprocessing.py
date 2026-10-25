"""Preprocessing pipeline for the BRFSS 2015 data (Project 1, Phase 4).

All column choices below come from the BRFSS 2015 codebook and were validated in the notebook
(Phase 4). Usage:

    xr_train, xr_test, kept_names = recode_data(x_train, x_test, col_names)
    nominal = column_indices(kept_names, NOMINAL)
    prep = fit_preprocessing(xr_train, TAU, nominal)        # statistics from training data only
    tx_train, tx_test = transform(xr_train, prep), transform(xr_test, prep)
"""
import numpy as np

# ---------------------------------------------------------------------------------------------
# Columns excluded a priori (4.1)
# ---------------------------------------------------------------------------------------------

# Interview process and survey design: they describe the survey, not the respondent's health
ADMIN = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
         "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT", "CTELNUM1",
         "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "QSTVER", "QSTLANG",
         "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT"]

# Missingness pattern deterministically reveals the target (codebook skip note)
LEAKAGE = ["HAREHAB1"]

# (Near-)linear functions of features already in the model, from the codebook definitions
REDUNDANT = ["WEIGHT2", "HEIGHT3", "HTIN4",                                   # -> WTKG3, HTM4, _BMI5
             "_AGEG5YR", "_AGE65YR", "_AGE_G",                                # -> _AGE80
             "MAXVO2_", "FC60_",                                              # functions of age and sex
             "ALCDAY5",                                                       # -> DROCDY3_
             "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",  # -> per-day versions
             "EXRACT11", "EXRACT21",                                          # -> METVL11_, METVL21_
             "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2", "STRENGTH",      # -> PAFREQ*_, PADUR*_, STRFREQ_
             "_PRACE1", "_MRACE1", "_RACEGR3", "_RACE_G1", "_RACEG21",        # -> _RACE
             "_EDUCAG", "_INCOMG", "CHILDREN",                                # -> EDUCA, INCOME2, _CHLDCNT
             "_DRDXAR1", "_TOTINDA", "_AIDTST3", "_LTASTH1", "_RFCHOL",       # same question as a raw variable
             "SMOKDAY2", "SMOKE100",                                          # -> _SMOKER3 dummies
             "_HISPANC",                                                      # -> _RACE = 8 dummy
             "_ASTHMS1",                                                      # = 2 - (_CASTHM1 - 1) + (ASTHMA3 - 1)
             "_PASTRNG",                                                      # -> _PAREC1 in {1, 3}
             "_MISFRTN", "_MISVEGN",                                          # sums of missingness indicators
             "LMTJOIN3", "ARTHDIS2", "ARTHSOCL",                              # -> _LMTACT1, _LMTWRK1, _LMTSCL1 with HAVARTH3
             "PA1MIN_", "PA1VIGM_",                                           # = PAMIN11_ + PAMIN21_, PAVIG11_ + PAVIG21_
             "NUMMEN", "NUMWOMEN",                                            # sum = NUMADULT
             "DRNKANY5", "_CHOLCHK", "_RFBING5",                              # implied by missingness indicators
             "_RFHYPE5", "_RFSMOK3", "_RFBMI5", "_RFSEAT2", "_RFSEAT3",       # groupings of one-hot variables
             "_PA150R2", "_PA300R2", "_PAINDX1", "_PA30021", "_PASTAE1",      # -> _PACAT1, _PAREC1
             "_FRUITEX", "_VEGETEX",                                          # = missingness of _FRUTSUM, _VEGESUM
             "_FRT16", "_FRTRESP", "_VEG23", "_VEGRESP"]                      # -> _FRUTSUM, _VEGESUM

# Month and year of the last flu shot / HIV test
DATES = ["FLSHTMY2", "HIVTSTD3"]

# Nominal with 53 levels: meaningless as a number, and its one-hot encoding gave no measurable gain
NOT_USED = ["_STATE"]

EXCLUDED = ADMIN + LEAKAGE + REDUNDANT + DATES + NOT_USED

# ---------------------------------------------------------------------------------------------
# Recoding of special codes (4.2)
# ---------------------------------------------------------------------------------------------

# Continuous variables: family rules must not be applied (values like 77 or 888 are real measures)
CONTINUOUS = ["_AGE80", "HTM4", "WTKG3", "_BMI5", "METVL11_", "METVL21_",
              "PADUR1_", "PADUR2_", "PAFREQ1_", "PAFREQ2_", "STRFREQ_",
              "_MINAC11", "_MINAC21", "PAMIN11_", "PAMIN21_",
              "PAVIG11_", "PAVIG21_", "_FRUTSUM", "_VEGESUM",
              "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_"]

# 7 / 77 means "don't know" even though the value just below is a valid answer (codebook)
FORCE_DK = ["RCSRLTN2",                                       # 1-6 valid, 7 = don't know
            "AVEDRNK2", "DRNK3GE5", "MAXDRNKS", "HHADULT"]     # 1-76 valid, 77 = don't know

# Complete rules (codes to NaN, codes to 0) for columns the family rules get wrong (codebook)
CUSTOM_RULES = {"DROCDY3_": ([900], []),     # drink-occasions per day x 100
                "_DRNKWEK": ([99900], []),   # drinks per week x 100
                "DIABAGE2": ([98, 99], []),  # age at diabetes diagnosis: 88 is a valid age
                "STRFREQ_": ([99], []),      # times per week (3 implied decimals): 99000 = don't know
                "PAFREQ1_": ([99], []),
                "PAFREQ2_": ([99], [])}

# ---------------------------------------------------------------------------------------------
# Final configuration (4.3-4.5)
# ---------------------------------------------------------------------------------------------

TAU = 0.8

# One-hot encoded: nominal variables and ordinal variables with non-monotone risk
NOMINAL = ["PERSDOC2", "BPHIGH4", "DIABETE3", "MARITAL", "RENTHOM1", "EMPLOY1", "_RACE",
           "_SMOKER3", "_PAREC1", "IMFVPLAC", "WHRTST10", "SXORIENT", "TRNSGNDR", "MSCODE",
           "CAREGIV1", "CHECKUP1", "SEATBELT", "USENOW3", "_BMI5CAT", "_CHLDCNT", "_PACAT1",
           "LASTSMK2", "ACTIN21_"]

# Candidates for winsorization (not applied in the final logistic model, see 4.5)
WINSOR = CONTINUOUS + ["DROCDY3_", "_DRNKWEK"]


def read_column_names(path):
    """Feature names from the header of a data csv file (the 'Id' column is dropped)."""
    with open(path) as f:
        return np.array(f.readline().strip().split(",")[1:])


def family_codes(values, force_dk=False):
    """Special codes of an integer questionnaire column, following the BRFSS conventions.
    Returns (codes mapped to NaN, codes mapped to 0). A 7/77/777 code means 'don't know'
    if the value just below it (6/76/776) is absent, or if force_dk is True."""
    m = values.max()
    if m <= 9:
        return [9] + ([7] if force_dk or 6 not in values else []), []
    if m <= 99:
        return [99] + ([77] if force_dk or 76 not in values else []), [88]
    return [999] + ([777] if force_dk or 776 not in values else []), [555, 888]


def recoding_rules(x_train, col_names):
    """Names of the kept columns and, for each, the codes mapped to NaN and to 0."""
    col_idx = {name: j for j, name in enumerate(col_names)}
    kept_names = np.array([name for name in col_names if name not in EXCLUDED])
    recode = {}
    for name in kept_names:
        col = x_train[:, col_idx[name]]
        values = np.unique(col[~np.isnan(col)])
        if name in CUSTOM_RULES:
            recode[name] = CUSTOM_RULES[name]
        elif name in CONTINUOUS:
            recode[name] = ([], [])
        else:
            recode[name] = family_codes(values, name in FORCE_DK)
    return kept_names, recode


def apply_recoding(x, col_names, kept_names, recode):
    """Return the kept columns of x, with special codes mapped to NaN or 0."""
    col_idx = {name: j for j, name in enumerate(col_names)}
    out = x[:, [col_idx[n] for n in kept_names]].copy()
    for k, name in enumerate(kept_names):
        nan_codes, zero_codes = recode[name]
        out[np.isin(out[:, k], nan_codes), k] = np.nan
        out[np.isin(out[:, k], zero_codes), k] = 0
    return out


def recode_data(x_train, x_test, col_names):
    """Exclusion and recoding of train and test data, with rules derived from the training data."""
    kept_names, recode = recoding_rules(x_train, col_names)
    return (apply_recoding(x_train, col_names, kept_names, recode),
            apply_recoding(x_test, col_names, kept_names, recode), kept_names)


def column_indices(kept_names, names):
    """Positions of the given column names in the recoded data."""
    return frozenset(int(np.where(kept_names == n)[0][0]) for n in names)


# ---------------------------------------------------------------------------------------------
# Feature construction (4.3): fit on training data, apply to any data
# ---------------------------------------------------------------------------------------------

def raw_features(x, prep):
    """Clipped and imputed numeric columns, one-hot nominal columns and missingness indicators."""
    xn = x[:, prep["num_cols"]]
    xn[:, prep["clip_pos"]] = np.clip(xn[:, prep["clip_pos"]], prep["lo"], prep["hi"])
    xn = np.where(np.isnan(xn), prep["medians"], xn)
    onehot = [(x[:, [c]] == cats).astype(float) for c, cats in zip(prep["cat_cols"], prep["categories"])]
    ind = np.isnan(x[:, prep["ind_cols"]]).astype(float)
    return np.hstack([xn] + onehot + [ind])


def fit_preprocessing(x, tau, nominal=frozenset(), continuous=frozenset(), indicators=True):
    """Learn the preprocessing on (recoded) training data x: columns with NaN fraction <= tau,
    winsorization of the continuous columns at the 1st/99th percentiles, median imputation of
    numeric columns, one-hot encoding of the nominal columns (first category dropped as reference),
    one indicator per distinct missingness pattern (if indicators is True), removal of constant features,
    standardization."""
    nan_frac = np.isnan(x).mean(axis=0)
    cols = np.where(nan_frac <= tau)[0]
    num_cols = np.array([c for c in cols if c not in nominal], dtype=int)
    cat_cols = np.array([c for c in cols if c in nominal], dtype=int)

    # Winsorization bounds of the continuous columns (positions within num_cols)
    clip_pos = np.array([i for i, c in enumerate(num_cols) if c in continuous], dtype=int)
    lo = np.nanpercentile(x[:, num_cols[clip_pos]], 1, axis=0)
    hi = np.nanpercentile(x[:, num_cols[clip_pos]], 99, axis=0)

    medians = np.nanmedian(x[:, num_cols], axis=0)
    categories = [np.unique(x[~np.isnan(x[:, c]), c])[1:] for c in cat_cols]

    # One missingness indicator per distinct pattern (identical indicators are redundant)
    with_nan = cols[nan_frac[cols] > 0]
    _, first = np.unique(np.isnan(x[:, with_nan]), axis=1, return_index=True)
    ind_cols = with_nan[np.sort(first)] if indicators else with_nan[:0]

    prep = {"num_cols": num_cols, "cat_cols": cat_cols, "clip_pos": clip_pos, "lo": lo, "hi": hi,
            "medians": medians, "categories": categories, "ind_cols": ind_cols}
    feats = raw_features(x, prep)
    std = feats.std(axis=0)
    prep["keep"] = std > 0                          # drop constant features
    prep["mean"] = feats[:, prep["keep"]].mean(axis=0)
    prep["std"] = std[prep["keep"]]
    return prep


def transform(x, prep):
    """Apply the learned preprocessing and add the bias column."""
    feats = raw_features(x, prep)[:, prep["keep"]]
    return np.c_[np.ones(len(x)), (feats - prep["mean"]) / prep["std"]]


def feature_names(prep, kept_names):
    """Names of the features produced by transform, in the same order (bias excluded)."""
    names = ([kept_names[c] for c in prep["num_cols"]]
             + [f"{kept_names[c]}={v:g}" for c, cats in zip(prep["cat_cols"], prep["categories"]) for v in cats]
             + [f"NaN({kept_names[c]})" for c in prep["ind_cols"]])
    return np.array(names)[prep["keep"]]
