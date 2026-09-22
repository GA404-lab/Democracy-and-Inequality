# Democracy, Inequality and Human Capital - Python
<p align="center">
<img width="1456" height="755" alt="449717ec-b593-4f1e-9b03-f85dca789b2a_1809x938" src="https://github.com/user-attachments/assets/0d24460f-f6ec-4f75-af3a-51f27177bb7a" />
<img width="1809" height="938" alt="3c26bad1-5dd6-43b1-b1a8-71933a8b9ba3_1809x938" src="https://github.com/user-attachments/assets/9fe4ff62-5629-4f16-b570-9b92e815e9a6" />
<img width="1456" height="755" alt="672df370-0fa4-451a-bcba-d36d5d68f06b_1809x938" src="https://github.com/user-attachments/assets/8c46fb24-1ac6-4eea-8366-d8155477d4af" />
</p>
 
*Gervon Alcide*


An exploratory analysis of the relationships between democracy, inequality, prosperity and human capital using World Bank data.

## Data Collection and Preparation

I collected country-level data for 2010–2022 directly from the World Bank API using Python’s `wbgapi` library.

The dataset includes:

- Gini coefficient
- Human Capital Index
- GDP per capita
- GDP per capita adjusted for purchasing power
- Voice and Accountability

I used a loop to retrieve and process each indicator, calculate each country’s average across the period, and remove the original yearly columns. I then merged the indicators by country and removed observations with missing values.

## Analysis

The complete analysis, findings and interpretation are available in my Substack article:

[**Democracy Does Not Automatically Make Countries Equal**](https://4lyte4.substack.com/p/democracy-does-not-automatically)

Alternatively you can read the pdf [here on github](Democracy-Inequality-analysis.pdf).

## Code

The complete Python analysis can be found in [`democracy_inequality.py`](democracy_inequality.py).
