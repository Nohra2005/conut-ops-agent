# 🍩 Conut AI Chief of Operations Agent
**Conut AI Engineering Hackathon — 12-Hour Challenge**

---

## 📌 Business Problem

Conut is a growing sweets and beverages brand operating across **4 branches** (Conut Main, Conut-Tyre, Conut Jnah, Main Street Coffee). Despite strong customer volume, the business faces several operational blind spots:

- **Revenue imbalance** across branches with no clear explanation
- **Seasonal demand spikes** are unplanned, leading to stock-outs or over-preparation
- **Coffee & milkshake products underperform** (~1.3% of revenue) despite a dedicated coffee concept
- **Delivery channel is underdeveloped** — only 6–79 delivery customers per branch, yet they spend 54% more per order
- **Staffing irregularities** — attendance logs show shifts as short as <1 minute and one employee logging 173+ hours/month
- **No combo strategy** — high-value item combinations exist in the data but are not actively promoted

The goal is to build an **AI-Driven Chief of Operations Agent** that surfaces insights, forecasts demand, optimizes staffing, and recommends growth actions — all powered by real transaction and attendance data.

---

## 🏗️ Approach & Architecture

### Data Sources
| File | Contents |
|------|----------|
| `REP_S_00136_SMRY.csv` | Revenue summary by division & menu channel |
| `REP_S_00194_SMRY.csv` | Tax & sales summary by branch |
| `REP_S_00461.csv` | Time & attendance logs |
| `REP_S_00502.csv` | Delivery orders by customer (line-item level) |
| `rep_s_00150.csv` | Customer orders with timestamps & totals |
| `rep_s_00191_SMRY.csv` | Sales by item and product group |
| `rep_s_00334_1_SMRY.csv` | Monthly sales by branch |
| `rep_s_00435_SMRY.csv` | Average sales by menu category |

### Agent Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User / Operator                  │
└────────────────────────┬────────────────────────────┘
                         │ natural language query
                         ▼
┌─────────────────────────────────────────────────────┐
│             AI Operations Agent (LLM Core)          │
│              powered by OpenClaw - OpenAi           │
└────┬──────────┬──────────┬──────────┬───────────────┘
     │          │          │          │        
     ▼          ▼          ▼          ▼
 Combo      Demand     Staffing   Expansion  Growth
 Optimizer  Forecaster Optimizer  Advisor
     │          │          │          │
     └──────────┴──────────┴──────────┘
                         │
                         ▼
            ┌────────────────────┐
            │   Data Layer       │
            │  (CSV → Pandas)    │
            └────────────────────┘
```

### Five Core Modules

1. **Combo Optimizer** — mines order patterns to identify high-value product bundles (e.g. Classic Chimney + milkshake + sauce) and, generates best promotions.
2. **Demand Forecaster** — uses historical monthly sales per branch to predict upcoming peaks (regression / time-series)
3. **Expansion Advisor** — evaluates branch-level KPIs to assess feasibility of a new location by evaluating the evolution of volume sales and, delivery/dine-in ratio.
4. **Shift Staffing Optimizer** — analyzes attendance logs to flag anomalies and recommend optimal shift coverage
5. **Coffee & Milkshake Growth Engine** — identifies upsell opportunities based on order co-occurrence and timing data

> **OpenClaw Integration** is the mandatory orchestration layer connecting all modules to the agent's reasoning loop.

---

## ▶️ How to Run


### 1. Clone & Setup
```bash
git clone https://github.com/Nohra2005/conut-ops-agent.git
cd conut-ops-agent
```
### 2. Create a Venv
```bash
python -m venv venv
venv/Sources/activate
```

### 3. Prerequisites
```bash
pip install -r requirements.txt
```

### 4. Run Docker
```bash
Docker compose build
Docker compose up 
```

### 5. Acces APIs
You can access the API documentationfor each service by 
http://localhost:800X/docs
Where X = 1->5


### OpenClaw
We initialized OpenClaw on an EC2 instance on AWS
We cloned the git into the instance
Created skills for each endpoints
Linked Telegram for easy communication
Proofs are found in the PDF

## 📊 Key Results & Recommendations

### Key Findings

| Finding | Detail |
|---------|--------|
| 🏆 Top Branch | Conut Jnah — 5.69B revenue, 5,045 customers |
| 💰 Highest Spend | Conut-Tyre — 1.64M avg per customer |
| 🚚 Delivery Premium | Delivery customers spend **54% more** (2.49M vs 1.62M) |
| ☕ Beverage Gap | Coffee & milkshakes = only ~1.3% of branch revenue |
| 📅 Seasonal Peak | Oct–Nov spike at most branches; Conut Jnah peaks in December |
| ⚠️ Staffing Alert | One employee logged 173.6h/month — audit required |

### Recommendations

**Immediate (0–4 weeks)**
- 🎁 Launch a "Classic Chimney Combo" bundle (chimney + milkshake + sauce) based on delivery order patterns
- 🔍 Audit attendance system — investigate impossible shift durations
- 🛵 Activate delivery at Conut Main — currently only 6 delivery customers despite high delivery spend potential

**Short-Term (1–3 months)**
- 📈 Deploy demand forecasting model to pre-staff and pre-stock for Oct–Dec peaks
- ☕ Integrate coffee upsell prompt at POS — target customers with chimney orders
- 🗓️ Implement shift-staffing regression to reduce over/understaffing costs

**Strategic (3–12 months)**
- 🗺️ Evaluate 5th branch location based on Jnah-style high-volume, delivery-accessible profile
- 🎯 Build loyalty program — delivery customers are highest LTV segment
- 🤖 Full OpenClaw integration for autonomous daily ops briefings

---

## 👥 Team
Built during the **Conut AI Engineering Hackathon** — 12 hours, real data, real impact.

Tatiana NOHRA, Roy BAROUDY, Marc EL NAWAR

*For questions about the data pipeline or agent architecture, refer to the inline code comments or the Executive Brief PDF.*