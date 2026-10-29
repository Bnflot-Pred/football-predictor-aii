import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Predictor AI", page_icon="⚽", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; }
[data-testid="stAppViewContainer"] { max-width: 100%; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 900px; }
h1, h2, h3, p, label, div { text-align: right; }
.stMetric { direction: rtl; }
div[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
.stButton button { width: 100%; border-radius: 10px; min-height: 3rem; }
.small-note { color: #666; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football Predictor AI")
st.subheader("نسخة ويب تجريبية — Elite v1.1")
st.info("هذه أداة تحليل إحصائي تجريبية وليست ضمانًا للفوز. هذه النسخة لا تجلب البيانات أو الأسعار تلقائيًا؛ أدخل القيم بنفسك.")

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def prediction(lh, la, max_goals=12):
    ph = [poisson_pmf(i, lh) for i in range(max_goals + 1)]
    pa = [poisson_pmf(i, la) for i in range(max_goals + 1)]
    matrix = [[ph[i] * pa[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    home = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i > j)
    draw = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i == j)
    away = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i < j)
    over25 = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i+j >= 3)
    btts = sum(matrix[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i >= 1 and j >= 1)
    scores = sorted([(i, j, matrix[i][j]) for i in range(max_goals + 1) for j in range(max_goals + 1)], key=lambda x: x[2], reverse=True)[:5]
    return home, draw, away, over25, btts, scores

def fair_probs(odds):
    raw = [1 / x for x in odds]
    total = sum(raw)
    return [x / total for x in raw], total - 1

with st.expander("١) أدخل بيانات المباراة", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        home = st.text_input("الفريق صاحب الأرض", "الفريق A")
        lh = st.slider("الأهداف المتوقعة للفريق الأول", 0.1, 4.5, 1.55, 0.05)
    with c2:
        away = st.text_input("الفريق الضيف", "الفريق B")
        la = st.slider("الأهداف المتوقعة للفريق الثاني", 0.1, 4.5, 1.10, 0.05)

h, d, a, over25, btts, scores = prediction(lh, la)

st.header("٢) احتمالات المباراة")
c1, c2, c3 = st.columns(3)
c1.metric(f"فوز {home}", f"{h:.1%}")
c2.metric("التعادل", f"{d:.1%}")
c3.metric(f"فوز {away}", f"{a:.1%}")

c4, c5 = st.columns(2)
c4.metric("أكثر من 2.5 هدف", f"{over25:.1%}")
c5.metric("الفريقان يسجلان", f"{btts:.1%}")

st.subheader("أكثر النتائج الصحيحة احتمالًا")
score_df = pd.DataFrame([
    {"النتيجة": f"{home} {i} - {j} {away}", "الاحتمال": f"{p:.2%}"}
    for i, j, p in scores
])
st.dataframe(score_df, hide_index=True, use_container_width=True)

st.header("٣) مقارنة أسعار السوق")
st.caption("أدخل أسعار 1-X-2 من شركة واحدة للمباراة نفسها. الأسعار الافتراضية للتجربة فقط.")
c1, c2, c3 = st.columns(3)
with c1:
    oh = st.number_input(f"فوز {home}", min_value=1.01, value=2.10, step=0.01)
with c2:
    od = st.number_input("تعادل", min_value=1.01, value=3.30, step=0.01)
with c3:
    oa = st.number_input(f"فوز {away}", min_value=1.01, value=3.60, step=0.01)

mkt, margin = fair_probs([oh, od, oa])
model = [h, d, a]
labels = [f"فوز {home}", "التعادل", f"فوز {away}"]
rows = []
for label, mp, qp, odds in zip(labels, model, mkt, [oh, od, oa]):
    rows.append({
        "السوق": label,
        "احتمال النموذج": f"{mp:.1%}",
        "احتمال السوق بعد إزالة الهامش": f"{qp:.1%}",
        "الفرق": f"{mp-qp:+.1%}",
        "العائد النظري المقدر": f"{(mp*odds-1):+.1%}"
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.write(f"هامش الشركة التقريبي قبل إزالة الهامش: **{margin:.1%}**")

best = max(range(3), key=lambda i: model[i])
best_ev_i = max(range(3), key=lambda i: model[i] * [oh, od, oa][i] - 1)
ev = model[best_ev_i] * [oh, od, oa][best_ev_i] - 1
st.subheader("٤) ملخص آلي")
st.write(f"النتيجة المفردة الأعلى احتمالًا هي **{labels[best]}** باحتمال {model[best]:.1%}. هذا لا يعني أن احتمالها أكبر من جميع النتائج الأخرى مجتمعة.")
if ev > 0:
    st.warning(f"يوجد فرق حسابي إيجابي وفق المدخلات على خيار **{labels[best_ev_i]}** (عائد نظري {ev:+.1%}). قد يكون هذا خطأً بسبب عدم دقة التقديرات، وليس ضمانًا أو توصية بالمراهنة.")
else:
    st.info("وفق القيم المدخلة، لا يظهر عائد نظري موجب بين خيارات 1-X-2.")

with st.expander("كيف تعمل النسخة وما حدودها؟"):
    st.markdown("""
    - تستخدم نموذج بواسون مستقلًا لتقدير عدد أهداف كل فريق.
    - الأهداف المتوقعة (λ) تدخل يدويًا، ولا تُحسب بعد من بيانات الفرق.
    - أسعار السوق تدخل يدويًا، ولا توجد تغذية مباشرة للأسعار.
    - إزالة الهامش تستخدم تطبيعًا تناسبيًا بسيطًا.
    - تغير الأسعار أو اختلافها لا يثبت التلاعب.
    - يجب اختبار النموذج على بيانات تاريخية خارج عينة التدريب قبل الثقة بنتائجه.
    """)

st.caption("Football Predictor AI • نموذج تجريبي • لا توجد ضمانات للنتائج أو الأرباح")
