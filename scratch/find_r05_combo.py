import pandas as pd
from itertools import combinations

# In our daily simulation for r05, recurring expenses included:
# Date 2025-11-06 (day 0): 706.37 (utilities)
# Date 2025-11-10 (day 4): 721.44 (healthcare)
# Date 2025-11-11 (day 5): 968.00 (debt_repayment)
# Date 2025-11-12 (day 6): 113.30 (cloud_storage)
# Date 2025-11-13 (day 7): 840.40 (family_support)
# Date 2025-12-02 (day 26): 4972.00 (rent)
# Date 2025-12-06 (day 30): 706.37 (utilities)
# Date 2025-12-10 (day 34): 721.44 (healthcare)
# Date 2025-12-11 (day 35): 968.00 (debt_repayment)
# Date 2025-12-12 (day 36): 113.30 (cloud_storage)
# Date 2025-12-13 (day 37): 840.40 (family_support)
# Date 2026-01-02 (day 57): 4972.00 (rent)
# Date 2026-01-06 (day 61): 706.37 (utilities)
# Date 2026-01-10 (day 65): 721.44 (healthcare)
# Date 2026-01-11 (day 66): 968.00 (debt_repayment)
# Date 2026-01-12 (day 67): 113.30 (cloud_storage)
# Date 2026-01-13 (day 68): 840.40 (family_support)
# Date 2026-02-02 (day 88): 4972.00 (rent)

items = [
    ("2025-11-06 utilities", 706.37),
    ("2025-11-10 healthcare", 721.44),
    ("2025-11-11 debt_repayment", 968.00),
    ("2025-11-12 cloud_storage", 113.30),
    ("2025-11-13 family_support", 840.40),
    ("2025-12-02 rent", 4972.00),
    ("2025-12-06 utilities", 706.37),
    ("2025-12-10 healthcare", 721.44),
    ("2025-12-11 debt_repayment", 968.00),
    ("2025-12-12 cloud_storage", 113.30),
    ("2025-12-13 family_support", 840.40),
    ("2026-01-02 rent", 4972.00),
    ("2026-01-06 utilities", 706.37),
    ("2026-01-10 healthcare", 721.44),
    ("2026-01-11 debt_repayment", 968.00),
    ("2026-01-12 cloud_storage", 113.30),
    ("2026-01-13 family_support", 840.40),
    ("2026-02-02 rent", 4972.00),
]

target_diff = 3758.23
print(f"Target difference: {target_diff}")

for r in range(1, len(items)):
    for combo in combinations(items, r):
        s = sum(x[1] for x in combo)
        if abs(s - target_diff) < 1.0:
            names = [x[0] for x in combo]
            print(f"MATCH: sum={s:.2f}, items={names}")
