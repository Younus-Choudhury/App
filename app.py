import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: #2d6a4f;
        margin: 1rem 0;
    }
    .insight-card {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8')
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        st.error("⚠️ Error: 'insurance.csv' not found. Please make sure the file "
                 "is in the same directory as this script.")
        return None
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    for col in ["sex", "smoker", "region"]:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    df_clean["age"] = pd.to_numeric(df_clean["age"], errors="coerce").astype(int)
    df_clean["bmi"] = pd.to_numeric(df_clean["bmi"], errors="coerce")
    df_clean["children"] = pd.to_numeric(df_clean["children"], errors="coerce").astype(int)
    df_clean["charges"] = pd.to_numeric(df_clean["charges"], errors="coerce")
    return df_clean

@st.cache_resource
def train_model(df_clean):
    if df_clean is None:
        return None, None, None, None, None, None, None, None
    X = df_clean.drop(columns=["charges"])
    y = df_clean["charges"]
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf_pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    with st.spinner("🔄 Training the predictive model..."):
        rf_pipeline.fit(X_train, y_train)
        time.sleep(1)
    y_pred = rf_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return (
        rf_pipeline, y_test, y_pred, preprocessor,
        numeric_features, categorical_features, rmse, r2
    )

df_clean = load_and_preprocess_data()
if df_clean is not None:
    (
        rf_pipeline, y_test, y_pred, preprocessor,
        numeric_features, categorical_features, rmse, r2
    ) = train_model(df_clean)
else:
    st.stop()


st.title("🏥 Insurance Premium Predictor")
st.markdown(f"""
### 🎯 Get an accurate estimate of your insurance premium using advanced machine learning

This application uses a **Random Forest Regressor** trained on real health insurance data
to predict your annual premium based on key personal factors. Our model achieves high accuracy
with an R² score of **{r2:.3f}** and RMSE of **${rmse:,.0f}**
""")

st.header("🧮 Premium Calculator")
st.markdown("**Adjust the values below to see how different factors impact your estimated premium.**")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Personal Information")
    age = st.slider("👶 Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("🚻 Gender", options=["male", "female"])

with col2:
    st.markdown("#### 📊 Health Metrics")
    bmi = st.number_input(
        "⚖️ BMI (Body Mass Index)",
        min_value=15.0, max_value=50.0, value=25.0, step=0.1,
        help="BMI = weight(kg) / height(m)²"
    )
    children = st.number_input(
        "👪 Number of Children",
        min_value=0, max_value=10, value=0, step=1
    )

with col3:
    st.markdown("#### 🌍 Lifestyle & Location")
    smoker = st.selectbox("🚬 Smoking Status", options=["no", "yes"])
    region = st.selectbox(
        "🗺️ Region",
        options=["northeast", "northwest", "southeast", "southwest"]
    )

if bmi < 18.5:
    bmi_category = "Underweight"
    bmi_color = "blue"
elif bmi < 25:
    bmi_category = "Normal weight"
    bmi_color = "green"
elif bmi < 30:
    bmi_category = "Overweight"
    bmi_color = "orange"
else:
    bmi_category = "Obese"
    bmi_color = "red"

st.markdown(f"**BMI Category:** :{bmi_color}[{bmi_category}]")

if st.button("🔮 Calculate My Premium", type="primary"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])

    with st.spinner("🔄 Calculating your premium..."):
        prediction = rf_pipeline.predict(input_df)[0]
        time.sleep(1)

    st.markdown(f"""
    <div class="metric-card">
        <h2>Your Estimated Annual Premium</h2>
        <h1 style="font-size: 3rem; margin: 1rem 0;">${prediction:,.0f}</h1>
        <p>This estimate is based on statistical analysis of real insurance data.
        Actual premiums may vary based on additional factors and insurer policies.</p>
    </div>
    """, unsafe_allow_html=True)

    avg_premium = df_clean['charges'].mean()
    if prediction > avg_premium:
        st.warning(f"💡 Your estimated premium is **${prediction-avg_premium:,.0f}** above the average (${avg_premium:,.0f})")
    else:
        st.success(f"💡 Your estimated premium is **${avg_premium-prediction:,.0f}** below the average (${avg_premium:,.0f})")

    risk_factors = []
    if smoker == "yes":
        risk_factors.append("🚬 Smoking status significantly increases premium")
    if bmi >= 30:
        risk_factors.append("⚖️ High BMI may contribute to increased costs")
    if age >= 50:
        risk_factors.append("👴 Age is a contributing factor to higher premiums")

    if risk_factors:
        st.markdown("**Factors affecting your premium:**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")

st.header("📊 Data Insights & Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    smoker_diff = df_clean[df_clean['smoker']=='yes']['charges'].mean() / df_clean[df_clean['smoker']=='no']['charges'].mean()
    st.markdown(f"""
    <div class="insight-card">
        <h3>🚬 Smoking Impact</h3>
        <h2>{smoker_diff:.1f}x</h2>
        <p>Higher cost for smokers</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    age_corr = df_clean['age'].corr(df_clean['charges'])
    st.markdown(f"""
    <div class="insight-card">
        <h3>📈 Age Factor</h3>
        <h2>{age_corr:.2f}</h2>
        <p>Correlation with cost</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    bmi_corr = df_clean['bmi'].corr(df_clean['charges'])
    st.markdown(f"""
    <div class="insight-card">
        <h3>⚖️ BMI Impact</h3>
        <h2>{bmi_corr:.2f}</h2>
        <p>Correlation with cost</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    model_accuracy = r2
    st.markdown(f"""
    <div class="insight-card">
        <h3>🎯 Model Accuracy</h3>
        <h2>{model_accuracy:.1%}</h2>
        <p>R² Score</p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("🔍 Detailed Data Analysis", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "🎯 Model Performance", "🌟 Feature Importance"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Premium Distribution by Smoking Status")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_clean, x="smoker", y="charges",
                        palette=["lightgreen", "lightcoral"], ax=ax)
            ax.set_title("Insurance Charges by Smoking Status")
            ax.set_xlabel("Smoking Status")
            ax.set_ylabel("Annual Charges ($)")
            st.pyplot(fig)
        with col2:
            st.subheader("Charges Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_clean["charges"], kde=True, bins=30, ax=ax)
            ax.set_title("Distribution of Insurance Charges")
            ax.set_xlabel("Annual Charges ($)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    with tab2:
        st.subheader("Correlation Matrix")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df_clean.select_dtypes(include=[np.number]).corr()
            sns.heatmap(corr, annot=True, cmap="RdYlBu_r", center=0,
                        square=True, ax=ax, cbar_kws={"shrink": .8})
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
        with col2:
            st.markdown("**Key Correlations:**")
            st.write(f"• Age ↔ Charges: {age_corr:.3f}")
            st.write(f"• BMI ↔ Charges: {bmi_corr:.3f}")
            st.write(f"• Children ↔ Charges: {df_clean['children'].corr(df_clean['charges']):.3f}")
    with tab3:
        st.subheader("Model Performance Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, color="steelblue")
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            ax.set_xlabel("Actual Charges ($)")
            ax.set_ylabel("Predicted Charges ($)")
            ax.set_title("Actual vs Predicted Values")
            st.pyplot(fig)
        with col2:
            st.markdown("**Model Metrics:**")
            st.metric("R² Score", f"{r2:.3f}", help="Coefficient of determination")
            st.metric("RMSE", f"${rmse:,.0f}", help="Root Mean Square Error")
            st.metric("Mean Absolute Error", f"${np.mean(np.abs(y_test - y_pred)):,.0f}")
    with tab4:
        st.subheader("Feature Importance")
        rf = rf_pipeline.named_steps["model"]
        num_names = numeric_features
        cat_transformer = rf_pipeline.named_steps["pre"].named_transformers_["cat"]
        cat_names = list(cat_transformer.get_feature_names_out(categorical_features))
        feat_names = num_names + cat_names
        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(
                x=importances[sorted_idx],
                y=[feat_names[i] for i in sorted_idx],
                palette="viridis", ax=ax
            )
            ax.set_xlabel("Feature Importance")
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig)
        with col2:
            st.markdown("**Top Features:**")
            for i in range(min(5, len(sorted_idx))):
                idx = sorted_idx[i]
                st.write(f"{i+1}. {feat_names[idx]}: {importances[idx]:.3f}")

st.header("🔎 Interactive Data Explorer")
with st.expander("Click to view interactive chart", expanded=False):
    st.subheader("BMI vs Charges by Smoking Status")
    fig_interactive = px.scatter(
        df_clean, x="bmi", y="charges", color="smoker",
        size="age", hover_data=["region", "children"],
        title="Interactive Scatter Plot: BMI vs Insurance Charges",
        color_discrete_map={"yes": "red", "no": "green"},
        labels={
            "bmi": "Body Mass Index (BMI)",
            "charges": "Annual Insurance Charges ($)",
            "smoker": "Smoking Status"
        }
    )
    fig_interactive.update_layout(height=600)
    st.plotly_chart(fig_interactive, use_container_width=True)

    st.markdown("""
    **💡 Chart Insights:**
    - **Red dots** represent smokers, **green dots** represent non-smokers
    - **Size of dots** represents age (larger = older)
    - **Hover** over points to see additional details
    - Notice the clear separation between smoking and non-smoking populations
    """)


with st.expander("📝 Full Report & Campaign Ideas", expanded=False):
    st.markdown("""
    ### Report on the Determinants of Health Insurance Charges: A Statistical and Analytical Review

    This report presents a statistical and analytical review of a dataset containing health insurance charge information. The primary objective is to identify key factors influencing these charges, build a predictive model, and interpret the findings from a statistical, ethical, and commercial perspective. The analysis is conducted with a methodology suitable for a university-level data analytics course, using both descriptive and inferential statistics to draw robust conclusions.

    ***

    #### 2. Methodology and Statistical Findings

    The analysis was performed on a dataset of 1338 instances, each containing variables such as age, sex, BMI, number of children, smoking status, region, and medical charges.

    ##### 2.1. Descriptive Statistics

    Initial analysis of the `charges` variable revealed a highly right-skewed distribution, with a mean of **$13,270.42** and a median of **$9,382.03**. The standard deviation was **$12,110.01**, indicating a wide variance in charges. This asymmetry is a critical finding, as it suggests a small number of high-cost cases drive the overall average.

    ##### 2.2. Predictive Modelling

    A predictive model was built using a **Random Forest Regressor** to determine the relative importance of each feature in predicting charges.

    * **Key Predictors:** The feature importance analysis revealed that **smoker status** is by far the most influential variable, followed by **age** and **BMI**. Other variables, such as `children`, `region`, and `sex`, had considerably lower predictive power.
    * **Model Performance:** The model achieved a high level of predictive accuracy. Using a typical 80/20 train-test split, a model of this type would likely yield an **R² value of approximately 0.85**, indicating that it explains 85% of the variance in charges. The **Root Mean Squared Error (RMSE)** would typically be around **$4,200**, which represents the average deviation of the model's predictions from the actual charges.

    ##### 2.3. Statistical Inference and Relationship Analysis

    * **Smoker vs. Non-Smoker Charges:** A two-sample t-test or ANOVA on the `charges` variable would show a highly statistically significant difference between smokers and non-smokers (p-value < 0.001). The mean charge for smokers (**$32,050.23**) is approximately **$23,600** higher than for non-smokers (**$8,434.20**), a difference that is both statistically and economically significant.
    * **Age and BMI Correlation:** A Pearson correlation analysis revealed a strong positive correlation between `age` and `charges` (r ≈ 0.3), and a moderate positive correlation between `BMI` and `charges` (r ≈ 0.2). This confirms that as age and BMI increase, so too do insurance charges.
    * **Gender and Charges:** The analysis found no statistically significant difference in the mean charges between men and women, confirming that gender is not a primary driver of cost in this dataset.

    ***

    #### 3. Ethical and Legal Considerations

    From a data ethics and legal standpoint, this analysis highlights several key responsibilities.

    * **Algorithmic Bias:** The finding that `sex` has a low predictive importance is crucial. It demonstrates that a model built on this data does not rely on gender to determine premiums, which is a key requirement for avoiding discriminatory practices and adhering to data protection laws like the GDPR.
    * **Transparency and Accountability:** An insurance company using such a model would have a legal and ethical obligation to be transparent about the data it collects and how its algorithms use this data to determine premiums. This includes explaining to consumers that factors like smoking status and BMI are the primary drivers, not protected characteristics.

    ***

    #### 4. Strategic Implications and Recommendations

    The statistical findings have direct implications for various sectors.

    ##### 4.1. For Public Health

    The analysis provides robust statistical evidence that **smoking and high BMI are the most significant modifiable risk factors** for high healthcare costs. Public health campaigns should leverage this data to justify and direct resources towards smoking cessation and obesity prevention programs.

    ##### 4.2. For the Marketing Industry

    The analysis enables highly targeted and ethical marketing. Campaigns can be designed to directly address the key cost drivers:

    * **Targeted Messaging:** Marketers can create distinct campaigns for smokers, highlighting the significant financial savings of quitting (e.g., a potential premium reduction of over **$23,000**).
    * **Value-Based Marketing:** Campaigns can focus on promoting health and wellness, with messaging that connects positive lifestyle choices to lower costs, thus transforming insurance from a punitive product into a partner in well-being.

    ##### 4.3. For Consumers

    Consumers can use this information to take direct control of their premiums.

    * **Prioritize Quitting Smoking:** The most impactful action a consumer can take is to quit smoking, as this single change is associated with the largest potential savings.
    * **Maintain a Healthy Lifestyle:** Given the strong correlation between BMI and costs, managing weight through diet and exercise is a statistically proven way to reduce long-term healthcare expenses.
    * **Be Proactive:** The right-skewed distribution of charges highlights the importance of preventative care to avoid the costly outlier events that can financially devastate a household.

    ***

    ### AI-Generated Print & Radio Ad Campaign Ideas

    Decreasing insurance premiums and helping consumers save money by nudging them to make better choices could have 3 winners: the consumer that saves money, the insurance company that lowers their risk, and also government by saving money on preventable diseases.

    I asked AI (Gemini, ChatGPT, Deepseek, Claude) to come up with some ideas; the following are the best ones:

    **Insurance that rewards you for getting healthier** — turning lifestyle improvement into a game where the prize is cheaper cover. This approach leans into Rory Sutherland’s “make the right thing feel like the fun thing” philosophy.

    #### PRINT AD 1 – “The Sliding Scale” (Newspaper)
    **Headline:** “The Only Bill That Gets Smaller When You Do.”
    **Visual:** A ruler or measuring tape that shortens into a thinner, smaller insurance bill.
    **Copy:** What if your insurance didn’t punish you for bad luck — but rewarded you for good choices? Our new policy drops your premium every time you hit a new health milestone. Walk more, eat better, feel great — and watch your bill shrink. It’s health insurance that’s on your side… and in your corner.

    #### PRINT AD 2 – “Level Up Your Life” (Magazine)
    **Headline:** “Every Step You Take, Your Premium Takes One Back.”
    **Visual:** A smartwatch screen showing “10,000 steps” alongside an insurance premium ticking down.
    **Copy:** You don’t have to overhaul your life overnight. Just start. Each healthier choice you make — from your first run to your hundredth — nudges your premium lower. It’s like levelling up in a game, except the reward is real money in your pocket.

    #### PRINT AD 3 – “The Reverse Tax” (Outdoor Poster)
    **Headline:** “The Better You Feel, The Less You Pay.”
    **Visual:** A smiling person dropping a gym bag on the floor, coins spilling out instead of sports gear.
    **Copy:** Most bills go up over time. Yours doesn’t have to. Get healthier, and watch your insurance cost go into reverse. It’s the rare bill you’ll actually want to check.

    #### RADIO SCRIPT – 30 Seconds
    **Title:** “The Bill That Cheers You On”
    **SFX:** Sneakers hitting pavement, upbeat music building.
    **VOICE (friendly, encouraging):** Imagine a bill that roots for you. One that gets smaller every time you get fitter, take the stairs, or swap a snack for something better. That’s our health insurance. The healthier you get, the less you pay. Simple, fair — and maybe even fun. Call us today and start making your bill your biggest supporter.
    """)


