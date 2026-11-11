"""
Python for Data Science — Beginner Course

Converted from python_data_science_beginners.ipynb.
Uses "# %%" cell markers: in VS Code, PyCharm or Spyder each block runs on its own,
like a notebook cell. The slide text appears as "# %% [markdown]" comment blocks.
Or run the whole file top to bottom:  python python_data_science_beginners.py

Differences from the notebook: tables are shown with print(), and each chart
opens in a window (close it to continue when running the whole file).
"""

# %% [markdown]
# # 🐍 Python for Data Science — From Zero to Insight
#
# ### A hands-on beginner course
#
# **By the end of today you will be able to:** load a real dataset, clean it, ask questions of it, and answer them with charts.
#
# | # | Module | Why it matters |
# |---|--------|----------------|
# | 1 | Data types | Everything in Python *is* something — know what |
# | 2 | Variables | Giving names to things so you can reuse them |
# | 3 | Data structures | Holding many values at once |
# | 4 | Conditional statements | Making decisions |
# | 5 | Loops (`for`, `while`) | Repeating work without copy-paste |
# | 6 | Functions | Packaging logic so you write it once |
# | 7 | pandas | Spreadsheets on steroids |
# | 8 | Exploratory Data Analysis | Interrogating data before trusting it |
# | 9 | Visualisation with seaborn | Seeing the story |
# | 🏁 | Capstone mini-project | Putting it all together |

# %% [markdown]
# ## How each module works
#
# Every module follows the same rhythm:
#
# 1. 🎯 **Focus** — what this concept is, in one sentence
# 2. 🕐 **When you use it** — real situations in data work
# 3. 🧪 **Live examples** — run them, change them, break them
# 4. 🔧 **Approaches & manipulation** — slicing, transforming, combining
# 5. ⚠️ **Common pitfalls** — mistakes everyone makes (so you don't have to)
# 6. ✋ **Your turn** — short exercises (solutions in the Appendix)
#
# **Running code:** click a cell and press `Shift + Enter`.
# **Golden rule:** if you're not sure what something does — *run it and look*.

# %%
# Setup: run this first. If anything fails, run:  pip install pandas seaborn matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 20)
print("✅ Ready! pandas", pd.__version__, "| seaborn", sns.__version__)

# %% [markdown]
# # Module 1 — Data Types 🧱
#
# 🎯 **Focus:** Every value in Python has a *type* that decides what you can do with it. You can add numbers, but you can't add a number to a word.
#
# 🕐 **When you use it:** Constantly. In data science, *wrong types are the #1 cause of silent bugs* — e.g. a "price" column read in as text, so `sort` puts `"100"` before `"20"`.
#
# | Type | Example | Used for |
# |------|---------|----------|
# | `int` | `42` | counts, IDs, ages |
# | `float` | `3.14` | prices, measurements, rates |
# | `str` | `"London"` | names, categories, free text |
# | `bool` | `True` / `False` | flags, filter results |
# | `NoneType` | `None` | "nothing here" / missing |

# %%
age = 29
price = 4.75
city = "London"
is_member = True
middle_name = None

for value in [age, price, city, is_member, middle_name]:
    print(repr(value), "→", type(value).__name__)

# %% [markdown]
# ### 🔧 Approach: converting types (casting)
#
# Data from files and users often arrives as **text**. Convert it before doing maths.

# %%
raw_price = "19.99"            # looks like a number... but it's a string
print(type(raw_price))

price = float(raw_price)       # str → float
quantity = int("3")            # str → int
print("Total:", price * quantity)

print(str(2026) + " budget")   # int → str so we can join text
print(int(7.9))                # float → int TRUNCATES (doesn't round!)
print(round(7.9))              # use round() if you want rounding
print(bool(0), bool(""), bool("hi"), bool(5))   # "truthiness"

# %% [markdown]
# ### 🔧 Approach: strings are sequences — so you can **slice** them
#
# Syntax: `text[start:stop:step]` — `start` is included, `stop` is **excluded**. Counting starts at **0**.
#
# ```
#  P   y   t   h   o   n
#  0   1   2   3   4   5
# -6  -5  -4  -3  -2  -1
# ```

# %%
word = "Python"
print(word[0])      # first character
print(word[-1])     # last character
print(word[0:3])    # 'Pyt'  (index 0, 1, 2)
print(word[2:])     # from index 2 to the end
print(word[:-2])    # everything except last 2
print(word[::2])    # every 2nd character
print(word[::-1])   # reversed!

# %%
# Practical: cleaning messy text — this is daily data-science work
customer = "   jane.DOE@Example.COM  "
clean = customer.strip().lower()
print(clean)
print("Username:", clean.split("@")[0])
print("Domain:  ", clean.split("@")[1])
print("Is gmail?", clean.endswith("gmail.com"))
print("Replace: ", clean.replace(".", "_"))

# f-strings: the best way to build text
total = 1234.5678
print(f"Total sales: £{total:,.2f}")      # thousands separator + 2 decimals
print(f"Growth: {0.0734:.1%}")            # percentage

# %% [markdown]
# ### ⚠️ Pitfalls
#
# * `"5" + 5` → **TypeError**. Python won't guess; convert first.
# * `0.1 + 0.2 == 0.3` is **False** (floats are approximations). Use `round()` or `math.isclose()`.
# * `int("3.5")` fails — go via `float` first: `int(float("3.5"))`.

# %%
print(0.1 + 0.2)
print(round(0.1 + 0.2, 10) == 0.3)
try:
    "5" + 5
except TypeError as e:
    print("TypeError:", e)

# %% [markdown]
# ## ✋ Your turn — Data types
#
# 1. `product_code = "UK-2026-COFFEE-0042"` — slice out the year (`"2026"`) and the number at the end (`"0042"`), then convert the number to an `int`.
# 2. Clean `"  MANCHESTER united "` into `"Manchester United"` (hint: `.strip()` and `.title()`).
# 3. Print `0.08765` as a percentage with 1 decimal place using an f-string.

# %%
product_code = "UK-2026-COFFEE-0042"
# YOUR CODE HERE

# %% [markdown]
# # Module 2 — Variables 🏷️
#
# 🎯 **Focus:** A variable is a **name tag** attached to a value, so you can refer to it later.
#
# 🕐 **When you use it:** Storing a loaded dataset (`df`), a threshold (`min_age = 18`), a result you'll reuse (`avg_sales`), configuration (`file_path`).
#
# **Naming rules & conventions**
# * letters, digits, underscores; can't start with a digit
# * `snake_case` is the Python style: `monthly_revenue` ✅ `MonthlyRevenue` ❌ `x2` 🤷
# * Be descriptive: `churn_rate` beats `cr`
# * Case-sensitive: `Sales` ≠ `sales`

# %%
monthly_revenue = 12_500        # underscores make big numbers readable
cost = 8_200
profit = monthly_revenue - cost
margin = profit / monthly_revenue
print(f"Profit £{profit:,} | margin {margin:.1%}")

# Re-assignment: the tag moves to a new value
profit = profit * 1.1
print("After 10% uplift:", profit)

# Shorthand updates
counter = 0
counter += 1      # same as counter = counter + 1
counter *= 10
print("counter =", counter)

# %%
# Handy tricks
a, b, c = 1, 2, 3            # multiple assignment
print(a, b, c)

a, b = b, a                  # swap in one line — no temp variable
print("swapped:", a, b)

first, *middle, last = [10, 20, 30, 40, 50]   # unpacking
print(first, middle, last)

# %% [markdown]
# ### ⚠️ The pitfall that bites every beginner: two names, one object
#
# For **mutable** objects (lists, dicts, DataFrames), `b = a` does **not** copy — both names point to the *same* object.

# %%
original = [1, 2, 3]
alias = original          # same object!
alias.append(99)
print("original:", original)     # 😱 changed too

safe_copy = original.copy()      # a genuine copy
safe_copy.append(100)
print("original:", original, "| copy:", safe_copy)
# Same idea in pandas: df2 = df.copy()

# %% [markdown]
# ## ✋ Your turn — Variables
#
# A shop sold `units = 340` items at `unit_price = 2.95` with `unit_cost = 1.10`.
# 1. Create variables for `revenue`, `total_cost` and `profit`.
# 2. Print one sentence with an f-string: *"Sold 340 units for £1,003.00 — profit £629.00"*.
# 3. Use one line to swap the values of `unit_price` and `unit_cost` (just to prove you can!).

# %%
units = 340
unit_price = 2.95
unit_cost = 1.10
# YOUR CODE HERE

# %% [markdown]
# # Module 3 — Data Structures 🗂️
#
# 🎯 **Focus:** Containers that hold **many values** in one variable.
#
# | Structure | Syntax | Ordered? | Changeable? | Duplicates? | Use it when… |
# |-----------|--------|----------|-------------|-------------|--------------|
# | **list** | `[1, 2, 3]` | ✅ | ✅ | ✅ | an ordered collection you'll add to / sort |
# | **tuple** | `(51.5, -0.1)` | ✅ | ❌ | ✅ | a fixed record, e.g. coordinates |
# | **dict** | `{"name": "Ana"}` | ✅ (insertion) | ✅ | keys unique | looking things up by a label |
# | **set** | `{"a", "b"}` | ❌ | ✅ | ❌ | finding unique values / overlaps |
#
# 💡 A pandas DataFrame is basically a **dict of columns**, where each column is like a **list**. Master these and pandas makes sense.

# %% [markdown]
# ## 3a. Lists — the workhorse
#
# 🕐 **When:** a column of values, a list of file names, results collected in a loop.

# %%
sales = [120, 340, 95, 410, 230, 180, 505]   # daily sales, Mon→Sun
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

print("First day:", days[0], "| Last day:", days[-1])
print("Weekdays:", days[:5])
print("Weekend: ", days[5:])
print("Every other day:", days[::2])
print("Midweek sales:", sales[1:4])
print("Reversed:", sales[::-1])
print(len(sales), sum(sales), min(sales), max(sales))

# %%
# Manipulating lists
scores = [72, 88, 65]
scores.append(91)            # add to end
scores.insert(0, 50)         # add at position
scores.extend([77, 83])      # add many
print("After adding:", scores)

scores.remove(65)            # remove by value
last = scores.pop()          # remove & return last
print("Popped:", last, "| now:", scores)

scores[0] = 55               # change by index
scores[1:3] = [70, 80]       # change a slice
print("Edited:", scores)

print("Sorted copy:", sorted(scores))            # original unchanged
scores.sort(reverse=True)                        # in place
print("Sorted in place (desc):", scores)
print("Is 91 in there?", 91 in scores, "| at index", scores.index(91))

# %% [markdown]
# ## 3b. Tuples — read-only records
#
# 🕐 **When:** values that belong together and shouldn't change — `(latitude, longitude)`, `(min, max)`, a function returning several values. `df.shape` is a tuple!

# %%
london = (51.5072, -0.1276)
lat, lon = london                       # unpacking
print(f"Lat {lat}, Lon {lon}")

try:
    london[0] = 0
except TypeError as e:
    print("Can't change a tuple:", e)

def min_max(values):
    return min(values), max(values)     # returns a tuple
low, high = min_max(sales)
print("Range:", low, "to", high)

# %% [markdown]
# ## 3c. Dictionaries — lookup by label
#
# 🕐 **When:** mapping codes to names (`{"GB": "United Kingdom"}`), storing a record (one row), config settings, counting things. JSON from APIs becomes dicts.

# %%
customer = {"name": "Ana", "age": 34, "city": "Leeds", "orders": [12.5, 40.0, 7.25]}

print(customer["name"])                    # access by key
print(customer.get("email", "no email"))  # safe access with default
customer["email"] = "ana@mail.com"         # add
customer["age"] += 1                       # update
del customer["city"]                       # remove
print(customer)

print("Keys:  ", list(customer.keys()))
print("Total spend:", sum(customer["orders"]))   # nested access

for key, value in customer.items():
    print(f"  {key:>6} → {value}")

# %%
# Practical: counting with a dict
transactions = ["card", "cash", "card", "online", "card", "online", "cash", "card"]
counts = {}
for method in transactions:
    counts[method] = counts.get(method, 0) + 1
print(counts)

# Dicts → DataFrame (sneak peek of pandas!)
print(pd.DataFrame({"method": list(counts.keys()), "count": list(counts.values())}))

# %% [markdown]
# ## 3d. Sets — uniqueness & overlaps
#
# 🕐 **When:** "how many *unique* customers?", "which IDs are in file A but not file B?" — a classic data-reconciliation task.

# %%
jan_customers = {"ana", "ben", "cai", "dee", "eve"}
feb_customers = {"cai", "dee", "fay", "gus"}

print("Unique in a list:", set(["a", "b", "a", "c", "b"]))
print("Both months (retained):", jan_customers & feb_customers)
print("Either month:          ", jan_customers | feb_customers)
print("Churned (Jan only):    ", jan_customers - feb_customers)
print("New in Feb:            ", feb_customers - jan_customers)

# %% [markdown]
# ## 3e. List comprehensions — the Pythonic one-liner
#
# `[expression for item in collection if condition]`

# %%
prices = [9.99, 24.50, 3.75, 120.00, 49.99]

with_vat = [round(p * 1.2, 2) for p in prices]
expensive = [p for p in prices if p > 20]
labels = ["high" if p > 40 else "low" for p in prices]
price_lookup = {f"item_{i}": p for i, p in enumerate(prices)}  # dict comprehension

print(with_vat)
print(expensive)
print(labels)
print(price_lookup)

# %% [markdown]
# ## ✋ Your turn — Data structures
#
# ```python
# temps = [14, 16, 19, 22, 25, 24, 20, 17, 15, 13, 11, 12]   # monthly avg °C, Jan→Dec
# ```
# 1. Slice out the summer months (Jun, Jul, Aug — indices 5–7) and the last quarter.
# 2. Make a dict `{"Jan": 14, "Feb": 16, ...}` using `zip(months, temps)` and `dict()`.
# 3. Using a set, find which of `["GBP","USD","EUR","USD","JPY","GBP"]` are unique currencies.
# 4. Use a list comprehension to convert all temps to Fahrenheit: `F = C * 9/5 + 32`.

# %%
temps = [14, 16, 19, 22, 25, 24, 20, 17, 15, 13, 11, 12]
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
# YOUR CODE HERE

# %% [markdown]
# # Module 4 — Conditional Statements 🔀
#
# 🎯 **Focus:** Run different code depending on whether something is `True` or `False`.
#
# 🕐 **When you use it:** business rules ("approve if score ≥ 650"), data validation ("skip if missing"), categorising ("high / medium / low"), handling edge cases.
#
# **Building blocks**
# * Comparisons: `==  !=  <  <=  >  >=`
# * Logic: `and`, `or`, `not`
# * Membership: `in`, `not in`
# * Identity: `is None`

# %%
credit_score = 680
income = 42_000

if credit_score >= 700:
    decision = "Approve"
elif credit_score >= 650 and income > 30_000:
    decision = "Approve with review"
elif credit_score >= 650:
    decision = "Refer"
else:
    decision = "Decline"

print(f"Score {credit_score}, income £{income:,} → {decision}")

# %% [markdown]
# 💡 **Order matters:** Python checks top to bottom and stops at the **first** `True`. Put the most specific / strictest condition first.
#
# **Try it live:** change `credit_score` to 720, 655, 600 and re-run.

# %%
# More patterns you'll use a lot
country = "FR"
if country in ["GB", "IE"]:
    region = "UK & Ireland"
elif country in {"FR", "DE", "ES", "IT"}:
    region = "EU core"
else:
    region = "Other"
print(region)

# One-line conditional (ternary)
temp = 23
print("warm" if temp > 20 else "cool")

# Checking for missing values
value = None
if value is None:
    print("Missing — impute or drop")

# Truthiness: empty things are False
items = []
if not items:
    print("No items to process")

# Chained comparison
age = 45
print("Working age" if 18 <= age < 67 else "Not working age")

# %% [markdown]
# ### ⚠️ Pitfalls
# * `=` assigns, `==` compares. `if x = 5:` is a SyntaxError.
# * `if x == "a" or "b":` is **always True** (because `"b"` is truthy). Write `if x in ("a", "b"):`.
# * Indentation *is* the syntax — 4 spaces under each `if`.

# %%
x = "z"
print(bool(x == "a" or "b"))   # True! — the bug
print(x in ("a", "b"))         # False — correct

# %% [markdown]
# ## ✋ Your turn — Conditionals
#
# Write a BMI classifier. Given `weight_kg = 70` and `height_m = 1.75`:
# * BMI = weight / height²
# * `< 18.5` → "Underweight", `< 25` → "Healthy", `< 30` → "Overweight", otherwise "Obese"
#
# Print: *"BMI 22.9 → Healthy"*. Then test with 3 other weights.

# %%
weight_kg = 70
height_m = 1.75
# YOUR CODE HERE

# %% [markdown]
# # Module 5 — Loops 🔁
#
# 🎯 **Focus:** Repeat a block of code without copy-pasting it.
#
# | | `for` loop | `while` loop |
# |-|-----------|--------------|
# | **Repeats…** | once for **each item** in a collection | **as long as** a condition is True |
# | **You know how many times?** | Yes (the length of the collection) | No — until something happens |
# | **Data-science examples** | process every file, every column, every row of results | retry until an API responds, iterate until a model converges, simulate until a target is hit |
#
# 👉 **Rule of thumb:** use `for` by default. Use `while` only when you don't know the number of repetitions in advance.

# %% [markdown]
# ## 5a. `for` loops

# %%
# Loop over a list
for city in ["London", "Paris", "Tokyo"]:
    print(f"Loading data for {city}...")

# Loop a fixed number of times with range(start, stop, step)
for year in range(2020, 2027, 2):
    print(year, end="  ")
print()

# Accumulate a result
sales = [120, 340, 95, 410, 230]
running_total = 0
for s in sales:
    running_total += s
print("Total:", running_total)

# %%
# The three helpers you'll use constantly
days  = ["Mon", "Tue", "Wed", "Thu", "Fri"]
sales = [120, 340, 95, 410, 230]

print("enumerate → index + value")
for i, day in enumerate(days, start=1):
    print(f"  Day {i}: {day}")

print("zip → walk two lists together")
for day, s in zip(days, sales):
    flag = "⭐" if s > 300 else ""
    print(f"  {day}: £{s} {flag}")

print(".items() → loop a dict")
stock = {"apples": 12, "pears": 0, "plums": 5}
for fruit, qty in stock.items():
    print(f"  {fruit}: {'OUT OF STOCK' if qty == 0 else qty}")

# %%
# Controlling the loop: break & continue
readings = [12.1, 11.8, None, 12.4, -999, 12.0, 11.9]

clean = []
for r in readings:
    if r is None:
        continue            # skip this one, carry on
    if r == -999:
        print("Sensor error code found — stopping")
        break               # exit the loop entirely
    clean.append(r)
print("Clean readings:", clean)

# Nested loops: every combination
for model in ["linear", "tree"]:
    for lr in [0.01, 0.1]:
        print(f"Train {model} with learning rate {lr}")

# %% [markdown]
# ## 5b. `while` loops

# %%
# How many years until savings double at 4.5% interest?
balance = 1_000
rate = 0.045
years = 0
while balance < 2_000:
    balance *= 1 + rate
    years += 1
print(f"Doubles after {years} years → £{balance:,.2f}")

# %%
# Simulating a retry loop (e.g. calling a flaky API)
import random
random.seed(3)

attempt = 0
max_attempts = 5
success = False
while not success and attempt < max_attempts:
    attempt += 1
    success = random.random() > 0.7      # pretend 30% chance of success
    print(f"Attempt {attempt}: {'✅ success' if success else '❌ failed'}")

if not success:
    print("Giving up after", max_attempts, "attempts")

# %% [markdown]
# ### ⚠️ Pitfalls
# * **Infinite `while` loop:** forgetting to update the condition variable. Always add a safety cap (`attempt < max_attempts`). In Jupyter, stop with ■ *Interrupt*.
# * **Changing a list while looping over it** → skipped items. Build a new list instead.
# * **Looping over DataFrame rows** is slow. In pandas, prefer *vectorised* operations (you'll see why in Module 7).

# %% [markdown]
# ## ✋ Your turn — Loops
#
# 1. **FizzBuzz (data edition):** loop through 1–20. Print `"Fizz"` for multiples of 3, `"Buzz"` for multiples of 5, `"FizzBuzz"` for both, else the number.
# 2. Given `prices = {"tea": 2.5, "coffee": 3.2, "cake": 4.0, "water": 1.0}`, loop through and build a new dict containing only items under £3.
# 3. **`while`:** a population of 1,000 grows 8% a year. How many years until it exceeds 5,000?

# %%
# YOUR CODE HERE

# %% [markdown]
# # Module 6 — Functions 🧩
#
# 🎯 **Focus:** A function is a **named, reusable block of code** that takes inputs (*parameters*) and gives back an output (`return`).
#
# 🕐 **When you use it:** As soon as you copy-paste code a second time. Cleaning steps, feature calculations, custom metrics, plotting helpers. Also every tool you use is a function: `len()`, `print()`, `pd.read_csv()`.
#
# **DRY** = *Don't Repeat Yourself*.
#
# ```python
# def function_name(param1, param2=default):
#     """What it does (docstring)."""
#     ...
#     return result
# ```

# %%
def add_vat(price, rate=0.20):
    """Return the price including VAT (default UK rate 20%)."""
    return round(price * (1 + rate), 2)

print(add_vat(10))              # uses default rate
print(add_vat(10, 0.05))        # positional
print(add_vat(rate=0, price=10))  # keyword arguments — order doesn't matter
print(add_vat.__doc__)          # docstrings power help(add_vat) too

# %%
# Functions can return several values (as a tuple)
def summarise(values):
    """Return count, mean and range of a list of numbers."""
    n = len(values)
    mean = sum(values) / n
    return n, round(mean, 2), max(values) - min(values)

count, avg, spread = summarise([120, 340, 95, 410, 230])
print(f"n={count}, mean={avg}, range={spread}")

# %%
# Combining everything so far: a reusable cleaning function
def clean_phone(raw):
    """Standardise a UK phone number to +44 format, or return None."""
    if raw is None:
        return None
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    if digits.startswith("44"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return "+44 " + digits

messy = ["07700 900123", "+44 7700-900456", "(0)7700900789", "12345", None]
print([clean_phone(p) for p in messy])

# %% [markdown]
# ### 🔧 Lambda functions — tiny, throwaway functions
#
# `lambda inputs: expression` — great for sorting keys and quick pandas transformations.

# %%
products = [("tea", 2.5), ("cake", 4.0), ("water", 1.0)]
print(sorted(products, key=lambda item: item[1]))            # sort by price
print(sorted(products, key=lambda item: item[1], reverse=True)[0])  # most expensive

square = lambda x: x ** 2
print(square(7))

# %% [markdown]
# ### ⚠️ Pitfalls
# * `print` ≠ `return`. A function that only prints gives back `None` — you can't use the result.
# * Variables created inside a function don't exist outside it (**scope**).
# * Never use a mutable default: `def f(items=[])` — use `items=None` instead.

# %%
def shout_print(text):
    print(text.upper())

def shout_return(text):
    return text.upper()

a = shout_print("hello")
b = shout_return("hello")
print("a =", a, "| b =", b)

# %% [markdown]
# ## ✋ Your turn — Functions
#
# 1. Write `bmi_category(weight_kg, height_m)` that **returns** the category from Module 4. Test it on 3 people.
# 2. Write `percent_change(old, new)` returning the % change rounded to 1 dp. Handle `old == 0` by returning `None`.
# 3. Write `describe_list(numbers)` that returns a **dict** with keys `min`, `max`, `mean`, `count`.

# %%
# YOUR CODE HERE

# %% [markdown]
# # Module 7 — pandas 🐼
#
# 🎯 **Focus:** pandas gives you the **DataFrame** — a table with labelled rows and columns — plus hundreds of tools to load, clean, reshape and summarise it.
#
# 🕐 **When you use it:** Almost every data-science project: reading CSV/Excel/SQL, cleaning, filtering, joining, aggregating, preparing data for models and charts.
#
# | Concept | Think of it as |
# |---------|----------------|
# | `Series` | one column (a labelled list) |
# | `DataFrame` | a whole table (a dict of Series) |
# | `index` | the row labels |
#
# **Everything you've learned so far shows up here:** types → `dtypes`, dicts → building DataFrames, slicing → `iloc`/`loc`, conditionals → boolean filters, functions → `apply`.

# %%
# A Series
s = pd.Series([120, 340, 95], index=["Mon", "Tue", "Wed"], name="sales")
print(s)
print("Tue sales:", s["Tue"], "| mean:", s.mean())

# %%
# Our practice dataset: one year of coffee-shop transactions (generated, so it works offline)
rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "order_id": range(1001, 1001 + n),
    "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "store": rng.choice(["Croydon", "Camden", "Brixton"], n, p=[0.4, 0.35, 0.25]),
    "product": rng.choice(["Latte", "Espresso", "Tea", "Muffin", "Sandwich"], n),
    "quantity": rng.integers(1, 5, n),
    "unit_price": np.nan,
    "loyalty_member": rng.choice([True, False], n, p=[0.3, 0.7]),
    "rating": rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[0.03, 0.07, 0.2, 0.35, 0.25, 0.1]),
})
price_map = {"Latte": 3.40, "Espresso": 2.60, "Tea": 2.20, "Muffin": 2.95, "Sandwich": 5.50}
df["unit_price"] = df["product"].map(price_map)
df.loc[rng.choice(n, 8, replace=False), "store"] = None              # some missing values
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True) # some duplicates

df.to_csv("coffee_sales.csv", index=False)   # save it so we can practise reading a file
df = pd.read_csv("coffee_sales.csv", parse_dates=["date"])
print(df.head())

# %% [markdown]
# ### 🔍 First look — always run these five

# %%
print("Shape (rows, cols):", df.shape)
print()
df.info()

# %%
print(df.describe())   # numeric summary

# %%
print(df.describe(include="object"))   # text columns: count, unique, most frequent

# %% [markdown]
# ## 🔧 Selecting data — columns

# %%
print(df["product"].head(3))   # one column → Series

# %%
print(df[["product", "quantity", "unit_price"]].head(3))   # list of columns → DataFrame (note the [[ ]])

# %% [markdown]
# ## 🔧 Slicing rows: `iloc` vs `loc`
#
# | | `iloc` | `loc` |
# |-|--------|-------|
# | selects by | **position** (0, 1, 2…) | **label** / condition |
# | end of slice | **excluded** (like lists) | **included** |
# | syntax | `df.iloc[rows, cols]` | `df.loc[rows, cols]` |

# %%
print(df.iloc[0:3, 0:4])   # first 3 rows, first 4 columns (by position)

# %%
print(df.iloc[-2:])   # last two rows

# %%
print(df.loc[0:3, ["store", "product"]])   # labels 0 to 3 INCLUSIVE, named columns

# %%
print(df.loc[df["store"] == "Croydon", ["date", "product", "quantity"]].head())   # condition + columns

# %% [markdown]
# ## 🔧 Filtering with conditions (boolean masks)
#
# Use `&` (and), `|` (or), `~` (not) — and **wrap each condition in brackets**.

# %%
mask = df["quantity"] >= 3
print(mask.head())                 # a Series of True/False...
print(df[mask].head(3))   # ...used to keep only True rows

# %%
big_croydon_lattes = df[(df["store"] == "Croydon") & (df["product"] == "Latte") & (df["quantity"] >= 3)]
food_or_low_rating = df[df["product"].isin(["Muffin", "Sandwich"]) | (df["rating"] <= 2)]
print(len(big_croydon_lattes), "big Croydon latte orders")
print(len(food_or_low_rating), "food orders or low ratings")

# .query() is a readable alternative
print(df.query("store == 'Camden' and quantity > 2 and loyalty_member").head(3))

# %% [markdown]
# ## 🔧 Creating & transforming columns
#
# **Vectorised** = the operation is applied to the whole column at once. Fast and readable — no loop needed.

# %%
df["revenue"] = df["quantity"] * df["unit_price"]                      # arithmetic
df["month"] = df["date"].dt.month_name().str[:3]                       # dates
df["weekday"] = df["date"].dt.day_name()
df["is_food"] = df["product"].isin(["Muffin", "Sandwich"])             # boolean
df["order_size"] = np.where(df["quantity"] >= 3, "large", "small")     # if/else on a column
df["price_band"] = pd.cut(df["unit_price"], bins=[0, 2.5, 3.5, 10],
                          labels=["budget", "mid", "premium"])          # binning

def satisfaction(r):                                                    # your own function
    if pd.isna(r):
        return "no rating"
    return "happy" if r >= 4 else "unhappy"
df["satisfaction"] = df["rating"].apply(satisfaction)

print(df[["product", "quantity", "revenue", "month", "order_size", "price_band", "satisfaction"]].head())

# %%
# Speed check: why we avoid loops in pandas
import time
t0 = time.perf_counter()
loop_result = [q * p for q, p in zip(df["quantity"], df["unit_price"])]
t1 = time.perf_counter()
vec_result = df["quantity"] * df["unit_price"]
t2 = time.perf_counter()
print(f"loop: {(t1-t0)*1e6:.0f} µs | vectorised: {(t2-t1)*1e6:.0f} µs  (gap grows hugely with millions of rows)")

# %% [markdown]
# ## 🔧 Sorting, counting, grouping
#
# `groupby` = **split** the data into groups → **apply** a calculation → **combine** the results. It answers *"... per ..."* questions: revenue **per** store, rating **per** product.

# %%
print(df.sort_values(["revenue", "date"], ascending=[False, True]).head(5))

# %%
print(df["product"].value_counts())

# %%
print(df.groupby("store")["revenue"].sum().sort_values(ascending=False))

# %%
summary = (
    df.groupby("product")
      .agg(orders=("order_id", "count"),
           units=("quantity", "sum"),
           revenue=("revenue", "sum"),
           avg_rating=("rating", "mean"))
      .round(2)
      .sort_values("revenue", ascending=False)
)
print(summary)

# %%
# Pivot table: store × product revenue
print(pd.pivot_table(df, values="revenue", index="store", columns="product", aggfunc="sum").round(0))

# %% [markdown]
# ## 🔧 Cleaning: missing values & duplicates

# %%
print("Missing per column:")
print(df.isna().sum()[df.isna().sum() > 0])
print("\nDuplicate rows:", df.duplicated().sum())

# %%
clean = df.copy()                                   # never overwrite your raw data
clean = clean.drop_duplicates()
clean["store"] = clean["store"].fillna("Unknown")   # fill categorical gaps
# alternative: clean = clean.dropna(subset=["rating"])  # drop rows instead (we keep them here)
clean = clean.rename(columns={"unit_price": "price"})
clean = clean.drop(columns=["price_band"])
print(clean.shape, "| missing stores left:", clean["store"].isna().sum())

# %%
# Joining tables: like VLOOKUP / SQL JOIN
stores = pd.DataFrame({
    "store": ["Croydon", "Camden", "Brixton"],
    "manager": ["Priya", "Tom", "Ama"],
    "seats": [40, 25, 30],
})
merged = clean.merge(stores, on="store", how="left")   # 'Unknown' store gets NaN manager
print(merged[["order_id", "store", "manager", "seats"]].sample(5, random_state=0))

# %% [markdown]
# ### ⚠️ Pitfalls
# * `df[df["a"] > 1 and df["b"] < 5]` → error. Use `&` and brackets: `df[(df["a"] > 1) & (df["b"] < 5)]`.
# * `SettingWithCopyWarning`: you're editing a slice. Use `.loc[...] = ...` or `.copy()`.
# * Most methods **return a new DataFrame** — `df.dropna()` alone changes nothing. Assign it: `df = df.dropna()`.
# * Check `dtypes` after loading — dates and numbers often arrive as text.

# %% [markdown]
# ## ✋ Your turn — pandas (use the `clean` DataFrame)
#
# 1. Show rows 10–14 and only the `date`, `store`, `revenue` columns — once with `iloc`, once with `loc`.
# 2. How many orders were from loyalty members at Brixton?
# 3. Which **weekday** has the highest total revenue?
# 4. For each store, what's the average rating and total units sold? Sort by average rating.
# 5. Add a column `discounted` that is 10% off `revenue` for loyalty members and full price otherwise (hint: `np.where`).

# %%
# YOUR CODE HERE

# %% [markdown]
# # Module 8 — Exploratory Data Analysis (EDA) 🔎
#
# 🎯 **Focus:** Getting to know a dataset *before* drawing conclusions or building models — its shape, quality, distributions and relationships.
#
# 🕐 **When you use it:** At the start of **every** project, and whenever new data arrives. EDA catches broken data early and tells you which questions are worth asking.
#
# ### The EDA checklist
# 1. **Structure** — rows, columns, types (`shape`, `info`, `dtypes`)
# 2. **Quality** — missing values, duplicates, impossible values
# 3. **Univariate** — each column on its own: distribution, centre, spread, outliers
# 4. **Bivariate** — pairs of columns: does X relate to Y?
# 5. **Segments** — does the pattern differ by group?
# 6. **Write down** what you found & what to investigate next

# %%
# We'll use seaborn's classic 'tips' dataset: restaurant bills & tips.
# (Needs internet the first time; falls back to a similar generated dataset if offline.)
try:
    tips = sns.load_dataset("tips")
    print("Loaded real 'tips' dataset")
except Exception:
    r = np.random.default_rng(0)
    m = 244
    size = r.choice([1, 2, 3, 4, 5, 6], m, p=[.02, .64, .15, .15, .02, .02])
    total = np.round(np.clip(r.gamma(4, 2.2, m) + size * 2.5, 3, 51), 2)
    tips = pd.DataFrame({
        "total_bill": total,
        "tip": np.round(np.clip(total * r.normal(0.155, 0.045, m), 1, 10), 2),
        "sex": r.choice(["Male", "Female"], m, p=[.64, .36]),
        "smoker": r.choice(["No", "Yes"], m, p=[.62, .38]),
        "day": r.choice(["Thur", "Fri", "Sat", "Sun"], m, p=[.25, .08, .36, .31]),
        "time": "Dinner", "size": size,
    })
    tips.loc[tips["day"].eq("Thur"), "time"] = "Lunch"
    print("Offline: using generated tips-like dataset")
print(tips.head())

# %% [markdown]
# ### Step 1–2: Structure & quality

# %%
print("Shape:", tips.shape)
print("\nTypes:\n", tips.dtypes)
print("\nMissing values:", tips.isna().sum().sum())
print("Duplicates:", tips.duplicated().sum())
print("Any negative bills?", (tips["total_bill"] <= 0).any())

# %% [markdown]
# ### Step 3: Univariate — one column at a time

# %%
print(tips.describe().round(2))

# %%
for col in tips.select_dtypes(exclude="number").columns:
    print(f"--- {col} ---")
    print(tips[col].value_counts(normalize=True).round(2).to_string(), "\n")

# %%
# Outliers with the IQR rule: below Q1 − 1.5×IQR or above Q3 + 1.5×IQR
q1, q3 = tips["tip"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = tips[(tips["tip"] < low) | (tips["tip"] > high)]
print(f"Normal tip range: £{low:.2f} – £{high:.2f} | outliers: {len(outliers)}")
print(outliers.sort_values("tip", ascending=False).head())

# %% [markdown]
# ### Step 4–5: Relationships & segments

# %%
tips["tip_pct"] = (tips["tip"] / tips["total_bill"] * 100).round(1)   # feature engineering!
print(tips[["total_bill", "tip", "size", "tip_pct"]].corr().round(2))

# %%
print(tips.groupby("day", observed=True)[["total_bill", "tip_pct"]].mean().round(2))

# %%
print(pd.crosstab(tips["day"], tips["time"], margins=True))

# %% [markdown]
# ### Step 6: Write it down 📝
#
# Good EDA ends in **plain-English findings**, e.g.:
# * *"Bigger bills get bigger tips (correlation ≈ 0.7), but tip **percentage** is roughly flat."*
# * *"Weekend dinners dominate the data — be careful generalising to weekday lunches."*
# * *"A handful of very large tips may be outliers — check before modelling."*
#
# Numbers tell you **what**; charts make it **obvious**. → Next module.

# %% [markdown]
# # Module 9 — Visualisation with seaborn 📊
#
# 🎯 **Focus:** seaborn turns a DataFrame into a statistical chart in one line. You pass `data=`, `x=`, `y=`, and optionally `hue=` (colour by group).
#
# 🕐 **When you use it:** During EDA (to spot patterns and problems fast) and when communicating results.
#
# ### Which chart do I need?
#
# | Question | Chart | seaborn |
# |----------|-------|---------|
# | How is one number distributed? | histogram / KDE | `histplot`, `kdeplot` |
# | How do groups compare on a number? | box / violin / bar | `boxplot`, `violinplot`, `barplot` |
# | How many in each category? | count bar | `countplot` |
# | Do two numbers move together? | scatter / regression | `scatterplot`, `regplot` |
# | Trend over time? | line | `lineplot` |
# | Correlation of many numbers? | heatmap | `heatmap` |
# | Everything vs everything? | pair grid | `pairplot` |

# %% [markdown]
# ### Distributions

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=tips, x="total_bill", bins=20, kde=True, ax=axes[0])
axes[0].set_title("Most bills are £10–£25, with a long right tail")
sns.histplot(data=tips, x="tip_pct", hue="time", element="step", ax=axes[1])
axes[1].set_title("Tip % by meal time")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Comparing groups

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
order = ["Thur", "Fri", "Sat", "Sun"]
sns.countplot(data=tips, x="day", order=order, ax=axes[0]).set_title("Visits per day")
sns.boxplot(data=tips, x="day", y="total_bill", order=order, ax=axes[1]).set_title("Bill size by day")
sns.barplot(data=tips, x="day", y="tip_pct", hue="smoker", order=order, ax=axes[2]).set_title("Avg tip % (bars = 95% CI)")
plt.tight_layout(); plt.show()

# %% [markdown]
# 💡 **How to read a boxplot:** the box is the middle 50% of values, the line is the median, whiskers show the typical range, dots are outliers.

# %% [markdown]
# ### Relationships

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", size="size", alpha=0.7, ax=axes[0])
axes[0].set_title("Bigger bills → bigger tips")
sns.regplot(data=tips, x="total_bill", y="tip", scatter_kws={"alpha": 0.4}, ax=axes[1])
axes[1].set_title("With a fitted trend line")
plt.tight_layout(); plt.show()

# %%
corr = tips.select_dtypes("number").corr()
plt.figure(figsize=(6, 4.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation heatmap"); plt.show()

# %%
sns.pairplot(tips, vars=["total_bill", "tip", "tip_pct"], hue="time", corner=True, height=2.3)
plt.show()

# %% [markdown]
# ### Small multiples & time series

# %%
# One chart per group with col= / row=
sns.relplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker", height=3.8, alpha=0.7)
plt.show()

# %%
# Back to our coffee data: daily revenue over time
daily = clean.groupby(["date", "store"], as_index=False)["revenue"].sum()
daily["revenue_7d"] = daily.groupby("store")["revenue"].transform(lambda s: s.rolling(7, min_periods=1).mean())
plt.figure(figsize=(12, 4))
sns.lineplot(data=daily[daily["store"] != "Unknown"], x="date", y="revenue_7d", hue="store")
plt.title("7-day rolling average revenue by store"); plt.ylabel("£ revenue"); plt.show()

# %% [markdown]
# ### 🎨 Making charts presentation-ready
# * **Title = the insight**, not the variable names ("Weekend bills are higher" > "total_bill by day")
# * Label axes with units; remove clutter
# * Use `order=` to sort categories meaningfully
# * Consistent colours: same group = same colour across charts

# %%
prod_rev = clean.groupby("product")["revenue"].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=prod_rev, x="revenue", y="product", color="#4C72B0")
ax.bar_label(ax.containers[0], fmt="£%.0f", padding=3)
ax.set(title="Sandwiches & lattes drive most revenue", xlabel="Total revenue (£)", ylabel="")
sns.despine(left=True); plt.tight_layout(); plt.show()

# %% [markdown]
# ## ✋ Your turn — seaborn
#
# Using `tips`:
# 1. Histogram of `tip` split by `sex` (`hue=`).
# 2. Boxplot of `tip_pct` by `size` (party size). What do you notice?
# 3. Using `clean` (coffee data), a `countplot` of `product` coloured by `store`.
# 4. **Challenge:** a bar chart of average `rating` per `weekday`, ordered Monday → Sunday, with an insight-led title.

# %%
# YOUR CODE HERE

# %% [markdown]
# # 🏁 Capstone mini-project — "Advise the coffee-shop owner"
#
# The owner of the three coffee shops asks you: **"Where should I focus next year?"**
#
# Work in pairs (30 min). Use everything from today:
#
# 1. **Load** `coffee_sales.csv` fresh with `pd.read_csv` (practise the full pipeline).
# 2. **Clean** it — duplicates, missing stores, types. Write a `clean_data(df)` **function** that returns the cleaned DataFrame.
# 3. **EDA** — which store, product, month and weekday earn the most? Do loyalty members spend more per order?
# 4. **Segment** — use a **loop** to print a one-line summary per store: *"Croydon: £X revenue, best product Y, avg rating Z"*.
# 5. **Visualise** — 2–3 charts with insight-led titles.
# 6. **Recommend** — 3 bullet points for the owner, each backed by a number or chart.
#
# 🏆 Present your top chart and your #1 recommendation in 60 seconds.

# %%
# Starter scaffold
def clean_data(raw):
    """Return a cleaned copy of the coffee sales data."""
    data = raw.copy()
    # YOUR CLEANING STEPS HERE
    return data

raw = pd.read_csv("coffee_sales.csv", parse_dates=["date"])
cafe = clean_data(raw)
print(cafe.head())

# %% [markdown]
# # 📋 Cheat sheet
#
# | Task | Code |
# |------|------|
# | Type / convert | `type(x)`, `int()`, `float()`, `str()` |
# | Slice | `seq[start:stop:step]`, `seq[-1]`, `seq[::-1]` |
# | f-string | `f"£{x:,.2f}"`, `f"{p:.1%}"` |
# | List | `.append() .extend() .pop() .sort() sorted() len()` |
# | Dict | `d[k]`, `d.get(k, default)`, `.items()` |
# | Set ops | `a & b`, `a | b`, `a - b` |
# | Comprehension | `[f(x) for x in xs if cond]` |
# | Loop helpers | `range()`, `enumerate()`, `zip()`, `break`, `continue` |
# | Function | `def f(a, b=1): return ...` / `lambda x: ...` |
# | Load / inspect | `pd.read_csv()`, `.head() .info() .describe() .shape` |
# | Select | `df["c"]`, `df[["a","b"]]`, `.iloc[r, c]`, `.loc[r, c]` |
# | Filter | `df[(df.a > 1) & (df.b == "x")]`, `.isin()`, `.query()` |
# | New column | `df["c"] = ...`, `np.where()`, `.apply()`, `.map()`, `pd.cut()` |
# | Summarise | `.value_counts()`, `.groupby().agg()`, `pd.pivot_table()` |
# | Clean | `.isna().sum()`, `.fillna()`, `.dropna()`, `.drop_duplicates()` |
# | Join | `df.merge(other, on="key", how="left")` |
# | Plot | `sns.histplot / boxplot / countplot / scatterplot / lineplot / heatmap / pairplot` |

# %% [markdown]
# # 🔑 Appendix — Solutions
#
# *Try first! Then compare. There's usually more than one correct answer.*

# %% [markdown]
# ### Data types

# %%
product_code = "UK-2026-COFFEE-0042"
year = product_code[3:7]
number = int(product_code[-4:])          # or product_code.split("-")[-1]
print(year, number)
print("  MANCHESTER united ".strip().title())
print(f"{0.08765:.1%}")

# %% [markdown]
# ### Variables

# %%
units, unit_price, unit_cost = 340, 2.95, 1.10
revenue = units * unit_price
total_cost = units * unit_cost
profit = revenue - total_cost
print(f"Sold {units} units for £{revenue:,.2f} — profit £{profit:,.2f}")
unit_price, unit_cost = unit_cost, unit_price
print(unit_price, unit_cost)

# %% [markdown]
# ### Data structures

# %%
temps = [14, 16, 19, 22, 25, 24, 20, 17, 15, 13, 11, 12]
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
print("Summer:", temps[5:8], "| Q4:", temps[-3:])
monthly = dict(zip(months, temps))
print(monthly)
print(set(["GBP", "USD", "EUR", "USD", "JPY", "GBP"]))
print([round(c * 9 / 5 + 32, 1) for c in temps])

# %% [markdown]
# ### Conditionals

# %%
for weight_kg in [50, 70, 85, 110]:
    height_m = 1.75
    bmi = weight_kg / height_m ** 2
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25:
        cat = "Healthy"
    elif bmi < 30:
        cat = "Overweight"
    else:
        cat = "Obese"
    print(f"BMI {bmi:.1f} → {cat}")

# %% [markdown]
# ### Loops

# %%
for i in range(1, 21):
    if i % 15 == 0:
        print("FizzBuzz", end=" ")
    elif i % 3 == 0:
        print("Fizz", end=" ")
    elif i % 5 == 0:
        print("Buzz", end=" ")
    else:
        print(i, end=" ")
print()

prices = {"tea": 2.5, "coffee": 3.2, "cake": 4.0, "water": 1.0}
cheap = {}
for item, p in prices.items():
    if p < 3:
        cheap[item] = p
print(cheap)            # or: {k: v for k, v in prices.items() if v < 3}

population, years = 1000, 0
while population <= 5000:
    population *= 1.08
    years += 1
print(f"{years} years → {population:,.0f}")

# %% [markdown]
# ### Functions

# %%
def bmi_category(weight_kg, height_m):
    bmi = weight_kg / height_m ** 2
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Healthy"
    elif bmi < 30:
        return "Overweight"
    return "Obese"

def percent_change(old, new):
    if old == 0:
        return None
    return round((new - old) / old * 100, 1)

def describe_list(numbers):
    return {"min": min(numbers), "max": max(numbers),
            "mean": sum(numbers) / len(numbers), "count": len(numbers)}

print(bmi_category(70, 1.75), bmi_category(95, 1.8), bmi_category(48, 1.7))
print(percent_change(200, 250), percent_change(0, 10))
print(describe_list([3, 8, 1, 9]))

# %% [markdown]
# ### pandas

# %%
print(clean.iloc[10:15][["date", "store", "revenue"]])
print(clean.loc[clean.index[10:15], ["date", "store", "revenue"]])

print("Brixton loyalty orders:", len(clean[(clean["store"] == "Brixton") & clean["loyalty_member"]]))

print(clean.groupby("weekday")["revenue"].sum().idxmax(), "is the best weekday")

print(clean.groupby("store").agg(avg_rating=("rating", "mean"), units=("quantity", "sum"))
           .sort_values("avg_rating", ascending=False).round(2))

clean["discounted"] = np.where(clean["loyalty_member"], clean["revenue"] * 0.9, clean["revenue"])
print(clean[["loyalty_member", "revenue", "discounted"]].head())

# %% [markdown]
# ### seaborn

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
sns.histplot(data=tips, x="tip", hue="sex", ax=axes[0, 0]).set_title("Tips by sex")
sns.boxplot(data=tips, x="size", y="tip_pct", ax=axes[0, 1]).set_title("Larger parties tip a slightly lower %")
sns.countplot(data=clean, x="product", hue="store", ax=axes[1, 0]).set_title("Orders by product & store")
week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.barplot(data=clean, x="weekday", y="rating", order=week, errorbar=None, ax=axes[1, 1])
axes[1, 1].set_title("Ratings are steady across the week")
axes[1, 1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()

# %% [markdown]
# # 🎉 Well done!
#
# **Where next?**
# * Practise on real data: [Kaggle datasets](https://www.kaggle.com/datasets), [data.gov.uk](https://www.data.gov.uk/)
# * pandas docs: *10 minutes to pandas*
# * seaborn gallery: https://seaborn.pydata.org/examples/
# * Next course: NumPy, scikit-learn & your first machine-learning model
#
# *Questions?* 🙋
