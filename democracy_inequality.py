import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wbgapi as wb

# Stats
from scipy import stats
import statsmodels.formula.api as smf

# Preprocessing & selection
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

# Boosting
from xgboost import XGBRegressor

# Pipeline
from imblearn.pipeline import Pipeline

# ==================================================================================

sns.set_theme(style="darkgrid", palette="rocket", font_scale=1.75)

wb.series.info(q="gini") #SI.POV.GINI
wb.series.info(q="human") #HD_HCIP_OVRL_TO #edu and health total
wb.series.info(q="gdp per capita") 
wb.series.info(q="voice", db=3) #GOV_WGI_VA.SC
#NY.GDP.PCAP.PP.CD # gdp per capita ppp
#NY.GDP.PCAP.CD normal gdp per cpita.
series = ["SI.POV.GINI", "HD_HCIP_OVRL_TO", "NY.GDP.PCAP.PP.CD", "NY.GDP.PCAP.CD", "GOV_WGI_VA.SC"]
dfs = dict.fromkeys(["gini", "eduhealth", "ppp", "percapita", "govtdem"], )

for i, s in zip(dfs, series):
    if i == "govtdem":
        dfs[i] = wb.data.DataFrame(
        s,
        time=range(2010, 2023),
        labels=True,
        db=3
        )
    else:
        dfs[i] = wb.data.DataFrame(
        s,
        time=range(2010, 2023),
        labels=True
)

years = []
for n in range(2010, 2023):
    years.append(f"YR{n}")

dfsc = dfs.copy()

for key in dfsc.keys():
    dfsc[key] = dfsc[key].reset_index(drop=False)
    dfsc[key][key] = dfsc[key][years].mean(axis=1)
    dfsc[key].drop(columns=years, inplace=True)

    print(dfsc[key].head())

dfsc["ppp"].head()
df = pd.merge(left=dfsc["ppp"], right=dfsc["gini"], on="Country", how="inner")
df = pd.merge(left=df, right=dfsc["eduhealth"], on="Country", how="inner")
df.drop(columns="economy", inplace=True)
df = pd.merge(left=df, right=dfsc["percapita"], on="Country", how="inner")
df.head()
df.drop(columns=["economy", "economy_x", "economy_y"], inplace=True)
df.head()
df = pd.merge(left=df, right=dfsc["govtdem"], on="Country", how="inner")
df.drop(columns="economy", inplace=True)
df.head()
df.isna().sum()
df.dropna(how="any", inplace=True)
df.head()
df.shape
df.dtypes
df.columns

# Distribution of metrics
fig, ax = plt.subplots(2,2)
sns.histplot(df["ppp"], ax=ax[0,1], kde=True)
sns.histplot(df["gini"], ax=ax[1, 1], kde=True, color="b")
sns.histplot(df["eduhealth"], ax=ax[0,0], kde=True, color="b")
sns.histplot(df["percapita"], ax=ax[1,0], kde=True)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2,1)
sns.histplot(df["govtdem"], ax=ax[0], kde=True)
sns.boxplot(x=df["govtdem"],ax=ax[1])
plt.show()

df.corr(numeric_only=True)
df.sort_values("govtdem", ascending=False)

# scatter for ppp and govtdem
fig, ax = plt.subplots()
sns.scatterplot(x=df["govtdem"], y=df["ppp"], hue=df["govtdem"])
ax.set_title("gdp per capita ppp v Democratic health")
plt.show()




#gini and govtdem

#ttest, but we have to check for normality and equal variances.
#starting with normality.
stats.normaltest(df["gini"]) # p = 0.001
stats.normaltest(df["govtdem"]) #p = 0.0001
#both not normal. now we do equal variance tests.
stats.levene(df["gini"], df["govtdem"]) #p = 0. not equal.
#testing for correlation
df[["gini", "govtdem"]].describe()
stats.spearmanr(df["ppp"], df["govtdem"]) #4 = 0.6 | p: 0

#reg plot
fig, ax = plt.subplots()
sns.regplot(x=df["govtdem"], y=df["gini"])
ax.set_title("gini v Democratic health")
plt.show()
smf.ols("gini ~ govtdem", data=df).fit().summary() #r2: 0.053. #p: 0.004

#rich countries democracy and inequality
dfrich = df[df["ppp"] > np.percentile(df["ppp"], 90)]
dfrich.reset_index(drop=True, inplace=True)
dfrich.sort_values("govtdem", ascending=True)
dfrich = dfrich[(dfrich["Country"] != "Qatar") & (dfrich["Country"] != "United Arab Emirates")]

fig, ax = plt.subplots(1,2)
sns.regplot(x=dfrich["govtdem"], y=dfrich["gini"], ax=ax[1])
sns.regplot(x=df["govtdem"], y=df["gini"], ax=ax[0])
ax[0].set_title("Entire Dataset")
ax[1].set_title("Rich Countries")
plt.show()

print(smf.ols("gini ~ govtdem", data=dfrich).fit().summary()) #r2: 0.47. p: 0.006
print(smf.ols("gini ~ govtdem", data=df).fit().summary()) #r2: 0.053. p: 0.004
print(stats.spearmanr(df["gini"], df["govtdem"])) #r = -0.24 | p: 0.002
print(stats.spearmanr(dfrich["gini"], dfrich["govtdem"])) #r = -0.25 | p: 0.37

#removing the outlier (United states) from dfrich's data
dfrich = dfrich[dfrich["Country"] != "United States"]
print(smf.ols("gini ~ govtdem", data=dfrich).fit().summary()) #r2: 0.13. p: 0.22

# looking at ppp and democracy below govtdem 65 and above.
df.head()
dfdemlow = df[df["govtdem"] < 65]
dfdemhigh = df[df["govtdem"] > 65]

fig, ax = plt.subplots(1, 2)
sns.regplot(
    x=dfdemlow["govtdem"],
    y=dfdemlow["ppp"],
    ax=ax[0]
)
sns.regplot(
    x=dfdemhigh["govtdem"],
    y=dfdemhigh["ppp"],
    ax=ax[1]
)
ax[0].set_title("<65 Democracy (low)")
ax[1].set_title(">65 Democracy (high)")
plt.show()

print(smf.ols("ppp ~ govtdem", data=dfdemlow).fit().summary()) #r2: 0.008. p:0.3
print(smf.ols("ppp ~ govtdem", data=dfdemhigh).fit().summary()) #r2: 0.6. p:0
# making robustness test
dfdemlow = df[df["govtdem"] < 60]
dfdemhigh = df[df["govtdem"] > 60]
print(smf.ols("ppp ~ govtdem", data=dfdemlow).fit().summary()) #r2: 0.0. p:0.9
print(smf.ols("ppp ~ govtdem", data=dfdemhigh).fit().summary()) #r2: 0.6. p:0

dfdemlow = df[df["govtdem"] < 70]
dfdemhigh = df[df["govtdem"] > 70]
print(smf.ols("ppp ~ govtdem", data=dfdemlow).fit().summary()) #r2: 0.011. p:0.2
print(smf.ols("ppp ~ govtdem", data=dfdemhigh).fit().summary()) #r2: 0.45. p:0

#setting the split back to what it was
dfdemlow = df[df["govtdem"] < 65]
dfdemhigh = df[df["govtdem"] > 65]

# doing the same high democracy low democracy thing but now with gini
fig, ax = plt.subplots(1, 2)
sns.regplot(
    x=dfdemlow["govtdem"],
    y=dfdemlow["gini"],
    ax=ax[0]
)
sns.regplot(
    x=dfdemhigh["govtdem"],
    y=dfdemhigh["gini"],
    ax=ax[1]
)
ax[0].set_title("<65 Democracy (low)")
ax[1].set_title(">65 Democracy (high)")
plt.show()

print(smf.ols("gini ~ govtdem", data=dfdemlow).fit().summary()) #r2: 0.01. p:0.2
print(smf.ols("gini ~ govtdem", data=dfdemhigh).fit().summary()) #r2: 0.2. p:0.001

# curventure check on ppp and democracy.
pipe = Pipeline(steps=[
    ("Poly", PolynomialFeatures(degree=2)),
    ("Regression", LinearRegression())
])
pipe.fit(df[["govtdem"]], df["ppp"])
ypredict = pipe.predict(df[["govtdem"]])
dftemp = pd.DataFrame({"govtdem": df["govtdem"], "Real PPP": df["ppp"], "Predict PPP": ypredict})
dftemp = dftemp.sort_values("govtdem", ascending=True)
dftemp = dftemp[~((dftemp["Real PPP"] > 60000) & (dftemp["govtdem"] < 70))]
dftemp.head()
#making the viz
fig, ax = plt.subplots()
sns.scatterplot(
    x=dftemp["govtdem"],
    y=dftemp["Real PPP"],
    ax=ax, 
    label="Actual PPP"
)
sns.lineplot(
    x=dftemp["govtdem"],
    y=dftemp["Predict PPP"],
    ax=ax,
    label="Predicted PPP",
    color="black"
)
ax.set_title("Dem & PPP Curventure")
plt.show()

df.head()

# Controlling for education and health
df.head()
print(smf.ols("gini ~ govtdem + eduhealth", data=df).fit().summary()) # coef: 0.0429 | p: 0.347 for democracy
print(smf.ols("eduhealth ~ govtdem", data=df).fit().summary()) # r: 0.4, coef: 2.2, p: 0
print(smf.ols("gini ~ govtdem + eduhealth", data=dfdemhigh).fit().summary()) # coef: 0.09 (similar to edu), p: 0.5
print(smf.ols("eduhealth ~ govtdem", data=dfdemhigh).fit().summary()) #r: 0.5, coef: 4, p: 0
print(smf.ols("eduhealth ~ govtdem", data=dfdemlow).fit().summary()) #r: 0.07, coef: 1.1, p: 0.05
# making a chart of gini eduhealth
fig, ax = plt.subplots()
sns.regplot(
    x=df["eduhealth"],
    y=df["gini"],
    ax=ax
)
ax.set_title("Gini by Education and Health")
plt.show()
#democracy
fig, ax = plt.subplots()
sns.regplot(
    x=df["govtdem"],
    y=df["eduhealth"],
    ax=ax
)
ax.set_title("Democracy  and Eduhealth in entire dataset")
plt.show()
# making chart of eduhealth against democracy health on high democracy and all countries
fig, ax = plt.subplots(1, 2)
sns.regplot(
    x=dfdemlow["govtdem"],
    y=dfdemlow["eduhealth"],
    ax=ax[0]
)
sns.regplot(
    x=dfdemhigh["govtdem"],
    y=dfdemhigh["eduhealth"],
    ax=ax[1],
    color="red"
)
ax[0].set_title("Low Democracies")
ax[1].set_title("High Democracies")
plt.show()

# redifining, finding next steps and making new questions
df.head()
# relationship between ppp and gini
df.corr(numeric_only=True, method="spearman")
print(smf.ols("ppp ~ gini", data=df).fit().summary())
fig, ax = plt.subplots()
sns.regplot(
    x=df["ppp"],
    y=df["gini"],
    ax=ax
)
plt.title("Inequality vs PPP")
plt.show()
# weird graph, lets test residuals.
model = LinearRegression()
model.fit(df[["gini"]], df["ppp"])
ypredict = model.predict(df[["gini"]])

dfy = df[["gini", "ppp"]]
dfy["yppp"] = ypredict
dfy["residuals"] = dfy["yppp"] - dfy["ppp"]
dfy.head()
fig, ax = plt.subplots()
sns.scatterplot(
    y=dfy["residuals"],
    x=dfy["gini"],
    ax=ax
)
plt.show()
# Failed the Heteroskedasticity.
# Testing cloud theory
dfphigh = df[df["ppp"] > 30000]
dfplow = df[df["ppp"] < 30000]
dfphigh.head()
print(smf.ols("gini ~ ppp", data=dfphigh).fit().summary()) #r: 0.007, coef: 0, p:0.6
print(smf.ols("gini ~ ppp", data=dfplow).fit().summary()) #r:0, coef: 0, p: 0.8

fig, ax = plt.subplots(1,2)
sns.regplot(
    x=dfplow["ppp"],
    y=dfplow["gini"],
    ax=ax[0]
)
sns.regplot(
    x=dfphigh["ppp"],
    y=dfphigh["gini"],
    ax=ax[1],
    color="red"
)
ax[0].set_title("Low PPP Countries")
ax[1].set_title("High PPP Countries")
plt.show()
# cloud theory proven true.

#**Does eduhealth predict gini**
df.head()
print(smf.ols("gini ~ eduhealth", data=df).fit().summary())

fig, ax = plt.subplots()
sns.regplot(
    x=df["eduhealth"],
    y=df["gini"],
    ax=ax
)
plt.show()

# **Classifying different levels of "prosperity"
fig, ax = plt.subplots(1,2)
sns.histplot(x=df["ppp"], ax=ax[0])
sns.histplot(x=df["gini"], ax=ax[1])
plt.show()

# making gini rank
def ginirank(x):
    if x < float(np.percentile(df["gini"], 33)):
        return "Equal"
    if x > float(np.percentile(df["gini"], 66)):
        return "Unequal"
    else:
        return "Mid"

df["gini_rank"] = df["gini"].apply(ginirank)
df.sample(30, random_state=1).sort_values("gini") # just making sure. everything looks good, no spillovers

# making ppp rank
def ppprank(x):
    if x < float(np.percentile(df["ppp"], 33)):
        return "Poor"
    if x > float(np.percentile(df["ppp"], 66)):
        return "Rich"
    else:
        return "Mid"
        
df["ppp_rank"] = df["ppp"].apply(ppprank)
df.head()

df["ppp_rank"].value_counts()
df["gini_rank"].value_counts()
df["P_Status"] = df["gini_rank"] + " and " + df["ppp_rank"]

# making bar charts
fig, ax = plt.subplots()
sns.barplot(x=df["P_Status"], y=df["govtdem"])
plt.xticks(size=15)
plt.show()

fig, ax = plt.subplots()
sns.barplot(x=df["P_Status"], y=df["eduhealth"])
plt.xticks(size=15)
plt.show()

df.head()
dftemp = df.groupby("P_Status")["Country"].agg("count")
dftemp.sort_values()
stats.chisquare(f_obs=dftemp)

# looking at whether wealth predict human capital.
smf.ols("eduhealth ~ ppp", data=df).fit().summary()
fig, ax = plt.subplots()
sns.regplot(y=df["eduhealth"], x=df["ppp"])
plt.show()
# making a polynomial regression model.
dftemp = df[["ppp", "eduhealth"]]
dftemp = dftemp[dftemp["ppp"] < 100000]

model = Pipeline(steps=[
    ("Poly", PolynomialFeatures(degree=2)),
    ("regression", LinearRegression())
])
model.fit(dftemp[["ppp"]], dftemp["eduhealth"])
ypredict = model.predict(dftemp[["ppp"]])

dftemp["eduhealthP"] = ypredict

print(mean_squared_error(dftemp["eduhealth"], ypredict)) #528
print(r2_score(dftemp["eduhealth"], ypredict)) # 0.8

fig, ax = plt.subplots()
sns.scatterplot(x=dftemp["ppp"], y=dftemp["eduhealth"])
sns.lineplot(x=dftemp["ppp"], y=dftemp["eduhealthP"])
plt.show()

### ML, predicting inequality

df.head()
dftemp = df.copy()
dftemp["ppp/edu"] = dftemp["ppp"] / dftemp["eduhealth"]
dftemp["percapita/ppp"] = dftemp["percapita"] / dftemp["ppp"]
dftemp["edu/govtdem"] = dftemp["eduhealth"] / dftemp["govtdem"]
dftemp.head()
Features = ["eduhealth", "percapita", "govtdem", "ppp/edu", "percapita/ppp", "edu/govtdem"]
X = dftemp[Features]

# trying different models
#Random Forest Regressor
pipe = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("rfr", RandomForestRegressor())
])

model = RandomizedSearchCV(
    estimator=pipe,
    param_distributions={
        "rfr__n_estimators": np.random.randint(low=50, high=300, size=20),
        "rfr__min_samples_split": np.random.randint(low=2, high=10, size=5),
        "rfr__max_depth": np.random.randint(low=2, high=50, size=20)},
    scoring="r2",
    cv=5,
    n_iter=50,
    random_state=5
)

model.fit(X, dftemp["gini"])

print(model.best_params_)
print(model.best_score_) #0.19

modelrfr = model.best_estimator_

#Gradient Booster
pipe = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("xgb", XGBRegressor())
])

model = RandomizedSearchCV(
    estimator=pipe,
    param_distributions={
        "xgb__n_estimators": np.random.randint(low=50, high=300, size=20),
        "xgb__learning_rate": np.random.uniform(low=0.1, high=1, size=5),
        "xgb__max_depth": np.random.randint(low=2, high=50, size=20)},
    scoring="r2",
    cv=5,
    n_iter=50,
    random_state=5
)

model.fit(X, dftemp["gini"])

print(model.best_params_)
print(model.best_score_) #-0.01

# terrible performance


#Neural Networks

pipe = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("mlp", MLPRegressor(max_iter=1000))
])


model = RandomizedSearchCV(
    estimator=pipe,
    param_distributions={
        "mlp__hidden_layer_sizes": [(50,), (100,), (50,50), (100,50)],
        "mlp__activation": ["relu", "tanh", "logistic"]},
    scoring="r2",
    cv=5,
    n_iter=20,
    random_state=5
)


model.fit(X, dftemp["gini"])

print(model.best_params_)
print(model.best_score_) # 0.24
modelmlp = model.best_estimator_

#ensemble model
modelmlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
modelrfr = RandomForestRegressor(max_depth=np.int32(4),
                                       min_samples_split=np.int32(9),
                                       n_estimators=np.int32(220))
model = StackingRegressor(
    estimators=[("mlp",modelmlp), ("rfr", modelrfr)],
    final_estimator=LinearRegression()
)

pipe = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("model", model)
])

score = cross_val_score(pipe, X, df["gini"], cv=5, scoring="r2")

print(score)

print(score.mean()) # 0.2

# The best model was the neural network, doing an r2 of 0.23
df2 = df[df["ppp"] < 100000]

df2.head()
print(smf.ols("eduhealth ~ ppp", data=df2).fit().summary()) 
print(smf.ols("eduhealth ~ gini", data=df2).fit().summary())
print(smf.ols("eduhealth ~ ppp + gini", data=df2).fit().summary())

model = Pipeline(steps=[
    ("poly", PolynomialFeatures(degree=2)),
    ("regress", LinearRegression())
])

model.fit(df2[["ppp"]], df2["eduhealth"])
ypredict = model.predict(df2[["ppp"]])

dftemp = df2[["eduhealth", "ppp"]]
dftemp["yeduhealth"] = ypredict
dftemp = dftemp[dftemp["ppp"] < 60000]
dftemp.head()

fig, ax = plt.subplots(1,2)
sns.scatterplot(y=dftemp["eduhealth"], x=dftemp["ppp"], ax=ax[0])
sns.lineplot(y=dftemp["yeduhealth"], x=dftemp["ppp"], ax=ax[0])
sns.regplot(y=df2["eduhealth"], x=df2["gini"], ax=ax[1])
plt.show()


# Checking different  relationshipos
print(smf.ols("gini ~ eduhealth", data=df2).fit().summary()) 
print(smf.ols("eduhealth ~ gini", data=df2).fit().summary())
print(smf.ols("eduhealth ~ ppp + gini", data=df2).fit().summary())
print(smf.ols("eduhealth ~ ppp + govtdem", data=df2).fit().summary())
print(smf.ols("gini ~ govtdem", data=df2).fit().summary())
print(smf.ols("gini ~ govtdem + ppp", data=df2).fit().summary())
