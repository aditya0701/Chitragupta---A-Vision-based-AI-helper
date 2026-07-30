import sys
sys.path.insert(0, r'd:\CV Exercise\AI_Chitragupt')
import server.agent as A

W = [("Agar - Wikipedia", "https://en.wikipedia.org/wiki/Agar", "seaweed"),
     ("Gelatin - Wikipedia", "https://de.wikipedia.org/wiki/Gelatin", "collagen")]
N = [("Agar guide", "https://cooking.example.com/agar", "how to use agar"),
     ("Substitutes", "https://food.example.org/subs", "agar swaps")]

orig = (A._search_mojeek, A._search_ddg_lite, A._search_ddg_instant)


def restore():
    A._search_mojeek, A._search_ddg_lite, A._search_ddg_instant = orig


# 1. Wikipedia stripped, non-Wikipedia kept
A._search_mojeek = lambda q, c: W + N
r = A.tool_web_search("agar")
print("[1 MIXED]"); print(r); print()
assert "wikipedia" not in r.lower(), "Wikipedia leaked through"
assert "cooking.example.com" in r

# 2. Provider returns ONLY Wikipedia -> must fall through to next provider
A._search_mojeek = lambda q, c: W
A._search_ddg_lite = lambda q, c: N
r = A.tool_web_search("agar")
print("[2 FALLTHROUGH]"); print(r); print()
assert "food.example.org" in r and "wikipedia" not in r.lower()

# 3. EVERY provider returns only Wikipedia -> must NOT read as "nothing exists"
A._search_mojeek = A._search_ddg_lite = A._search_ddg_instant = lambda q, c: W
r = A.tool_web_search("agar")
print("[3 ALL EXCLUDED]"); print(r); print()
assert "No web search results found" not in r, "REGRESSION: exclusion phrased as absence"
assert "NOT evidence" in r

# 4. Genuinely empty still reads as empty (not conflated with exclusion)
A._search_mojeek = A._search_ddg_lite = A._search_ddg_instant = lambda q, c: []
r = A.tool_web_search("zzqq")
print("[4 TRULY EMPTY]"); print(r); print()
assert "No web search results found" in r and "weak evidence" in r

# 5. fetch_page refuses an excluded domain without a round trip
print("[5 FETCH]", A.tool_fetch_page("https://en.wikipedia.org/wiki/Agar")[:110])
assert "configured not" in A.tool_fetch_page("en.wikipedia.org/wiki/Agar")

restore()
print("\nALL EXCLUSION ASSERTIONS PASSED")
