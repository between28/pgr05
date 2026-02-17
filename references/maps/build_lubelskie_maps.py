import re
import unicodedata
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from pypdf import PdfReader
from shapely.geometry import Point, LineString

BASE = Path('references/maps')
BASE.mkdir(parents=True, exist_ok=True)

GADM_PATH = BASE / 'gadm41_POL_2.json'
GUS_XLSX = BASE / 'gus_unemployment_dec2025.xlsx'

# 1) Load and prepare Lubelskie county geometry
powiat = gpd.read_file(GADM_PATH)
lub = powiat[powiat['NAME_1'] == 'Lubelskie'].copy()
lub = lub.to_crs(4326)

# 2) Extract unemployment (Dec 2025) from GUS table
raw = pd.read_excel(GUS_XLSX, sheet_name='Tabl.1a', header=None)
sub = raw[raw[0].astype(str).str.zfill(2) == '06'][[0, 1, 2, 3, 4]].copy()
sub.columns = ['woj', 'pow', 'powiat_name_pl', 'unemployed_thousand', 'unemployment_rate']
sub = sub[sub['pow'].astype(str).str.zfill(2) != '00'].copy()
sub['powiat_name_pl'] = sub['powiat_name_pl'].astype(str).str.strip()
sub['unemployment_rate'] = pd.to_numeric(sub['unemployment_rate'], errors='coerce')

powiat_to_gadm = {
    'bialski': 'BiałaPodlaska',
    'biłgorajski': 'Biłgoraj',
    'chełmski': 'Chełm',
    'hrubieszowski': 'Hrubieszów',
    'janowski': 'JanówLubelski',
    'krasnostawski': 'Krasnystaw',
    'kraśnicki': 'Kraśnik',
    'lubartowski': 'Lubartów',
    'lubelski': 'Lublin',
    'łęczyński': 'Łęczna',
    'łukowski': 'Łuków',
    'opolski': 'OpoleLubelskie',
    'parczewski': 'Parczew',
    'puławski': 'Puławy',
    'radzyński': 'RadzyńPodlaski',
    'rycki': 'Ryki',
    'świdnicki': 'Świdnik',
    'tomaszowski': 'TomaszówLubelski',
    'włodawski': 'Włodawa',
    'zamojski': 'Zamość',
    'm. biała podlaska': 'BiałaPodlaska(City)',
    'm. chełm': 'Chełm(City)',
    'm. lublin': 'Lublin(City)',
    'm. zamość': 'Zamość(City)',
}
sub['powiat_key'] = sub['powiat_name_pl'].str.lower()
sub['NAME_2'] = sub['powiat_key'].map(powiat_to_gadm)

unemp = sub[['powiat_name_pl', 'NAME_2', 'unemployed_thousand', 'unemployment_rate']].dropna(subset=['NAME_2']).copy()
unemp.to_csv(BASE / 'lubelskie_unemployment_dec2025.csv', index=False, encoding='utf-8')

# 3) Parse NOSG monthly PDFs for crossing-level traffic
nosg_url = 'https://nadbuzanski.strazgraniczna.pl/nos/komenda/granice/statystyki/2025'
html = requests.get(nosg_url, timeout=40).text
links = re.findall(r'href="([^"]+\.pdf)"', html)
links = [l for l in links if 'ruchdointernetu' in l.lower() or re.search(r'(styczen|luty|marzec|kwiecien|maj|czerwiec|lipiec|sierpien|wrzesien|pazdziernik|listopad|grudzien)2025', l, re.I)]
pdf_urls = []
for l in links:
    u = l if l.startswith('http') else 'https://nadbuzanski.strazgraniczna.pl' + l
    if u not in pdf_urls:
        pdf_urls.append(u)

month_map = {
    'styczen': 1, 'luty': 2, 'marzec': 3, 'kwiecien': 4, 'maj': 5, 'czerwiec': 6,
    'lipiec': 7, 'sierpien': 8, 'wrzesien': 9, 'pazdziernik': 10, 'listopad': 11, 'grudzien': 12
}

crossings = {
    'dorohusk_drogowe': 'Dorohusk road',
    'dorohusk_kolejowe': 'Dorohusk rail',
    'zosin_drogowe': 'Zosin road',
    'hrubieszow_kolejowe': 'Hrubieszow rail',
    'dolhobyczow_drogowe': 'Dolhobyczow road',
    'hrebenne_drogowe': 'Hrebenne road',
    'hrebenne_kolejowe': 'Hrebenne rail',
    'swidnik_lotnicze': 'Swidnik airport',
}

coords = {
    'Dorohusk road': (23.803, 51.154),
    'Dorohusk rail': (23.799, 51.157),
    'Zosin road': (24.047, 50.912),
    'Hrubieszow rail': (23.895, 50.802),
    'Dolhobyczow road': (24.050, 50.586),
    'Hrebenne road': (23.597, 50.249),
    'Hrebenne rail': (23.592, 50.251),
    'Swidnik airport': (22.694, 51.231),
}


def norm(s: str) -> str:
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace('ł', 'l').replace('ś', 's').replace('ż', 'z').replace('ź', 'z').replace('ć', 'c').replace('ń', 'n').replace('ą', 'a').replace('ę', 'e').replace('ó', 'o')
    s = s.replace('(', ' ').replace(')', ' ').replace('-', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s


def parse_tokens_with_ranges(tokens, ranges, check_fn=None):
    n = len(tokens)
    solutions = []

    def dfs(i, col, acc):
        if col == len(ranges):
            if i == n:
                if check_fn is None or check_fn(acc):
                    solutions.append(acc[:])
            return
        lo, hi = ranges[col]
        for ln in (1, 2, 3):
            if i + ln > n:
                continue
            val = int(''.join(tokens[i:i + ln]))
            if lo <= val <= hi:
                acc.append(val)
                dfs(i + ln, col + 1, acc)
                acc.pop()

    dfs(0, 0, [])
    return solutions[0] if solutions else None


PERSON_RANGES = [
    (0, 600000),
    (0, 600000),
    (0, 1200000),
    (0, 60000),
    (0, 60000),
    (0, 120000),
    (0, 1300000),
]

TRANSPORT_RANGES = [
    (0, 20000),
    (0, 120000),
    (0, 4000),
    (0, 120000),
    (0, 4000),
    (0, 4000),
    (0, 3000),
    (0, 300000),
]


def person_consistency(v):
    # [fa, fd, ft, pa, pd, pt, total]
    return (v[0] + v[1] == v[2]) and (v[3] + v[4] == v[5]) and (v[2] + v[5] == v[6])


def transport_consistency(v):
    # [buses, cars, motorcycles, trucks, passenger_trains, freight_trains, other, total]
    return sum(v[:7]) == v[7]

monthly_rows = []
for u in pdf_urls:
    month = None
    low = u.lower()
    for k, v in month_map.items():
        if k in low:
            month = v
            break
    if month is None:
        continue

    text = '\n'.join((p.extract_text() or '') for p in PdfReader(BytesIO(requests.get(u, timeout=50).content)).pages)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    norm_lines = [norm(ln) for ln in lines]

    split_idx = None
    for i, nln in enumerate(norm_lines):
        if nln.startswith('2. ruch graniczny srodkow transportu'):
            split_idx = i
            break

    if split_idx is None:
        person_section = list(zip(lines, norm_lines))
        transport_section = []
    else:
        person_section = list(zip(lines[:split_idx], norm_lines[:split_idx]))
        transport_section = list(zip(lines[split_idx:], norm_lines[split_idx:]))

    person_hits = {}
    transport_hits = {}

    for key in crossings.keys():
        tag = key.replace('_', ' ')

        for ln, nln in person_section:
            if tag in nln:
                nums = parse_tokens_with_ranges(re.findall(r'\d+', ln), PERSON_RANGES, person_consistency)
                if nums is not None:
                    person_hits[key] = nums
                    break

        for ln, nln in transport_section:
            if tag in nln:
                nums = parse_tokens_with_ranges(re.findall(r'\d+', ln), TRANSPORT_RANGES, transport_consistency)
                if nums is not None:
                    transport_hits[key] = nums
                    break

    for key, label in crossings.items():
        pnums = person_hits.get(key)
        tnums = transport_hits.get(key)
        monthly_rows.append({
            'month': month,
            'crossing_key': key,
            'crossing_label': label,
            'persons_total': pnums[6] if pnums else None,
            'persons_foreign_total': pnums[2] if pnums else None,
            'trucks_total': tnums[3] if tnums else None,
            'transport_total': tnums[7] if tnums else None,
            'source_pdf': u,
        })

monthly = pd.DataFrame(monthly_rows)
monthly = monthly.sort_values(['crossing_label', 'month'])
monthly.to_csv(BASE / 'nosg_crossings_monthly_2025.csv', index=False, encoding='utf-8')

annual = monthly.groupby(['crossing_key', 'crossing_label'], dropna=False)[['persons_total', 'trucks_total', 'transport_total']].sum(min_count=1).reset_index()
annual['lon'] = annual['crossing_label'].map(lambda x: coords.get(x, (None, None))[0])
annual['lat'] = annual['crossing_label'].map(lambda x: coords.get(x, (None, None))[1])
annual.to_csv(BASE / 'nosg_crossings_annual_2025.csv', index=False, encoding='utf-8')

# 4) Build map data
map_df = lub.merge(unemp[['NAME_2', 'unemployment_rate']], on='NAME_2', how='left')

priority_name2 = {
    'Chełm', 'Hrubieszów', 'TomaszówLubelski', 'Włodawa', 'Krasnystaw',
    'Chełm(City)', 'Lublin(City)'
}
map_df['priority'] = map_df['NAME_2'].isin(priority_name2)

pts = annual.dropna(subset=['lat', 'lon']).copy()
pts = gpd.GeoDataFrame(pts, geometry=[Point(xy) for xy in zip(pts['lon'], pts['lat'])], crs='EPSG:4326')
major_labels = ['Dorohusk road', 'Zosin road', 'Dolhobyczow road', 'Hrebenne road', 'Swidnik airport']
major_pts = pts[pts['crossing_label'].isin(major_labels)].copy()

# Lublin city centroid for corridor lines
lublin_city = map_df[map_df['NAME_2'] == 'Lublin(City)']
if len(lublin_city) > 0:
    lc = lublin_city.geometry.iloc[0].centroid
else:
    lc = Point(22.57, 51.25)

corridor_targets = ['Dorohusk road', 'Hrebenne road', 'Zosin road', 'Dolhobyczow road']
line_geoms = []
for lab in corridor_targets:
    r = pts[pts['crossing_label'] == lab]
    if len(r) > 0:
        p = r.geometry.iloc[0]
        line_geoms.append({'corridor': lab, 'geometry': LineString([lc, p])})
lines = gpd.GeoDataFrame(line_geoms, crs='EPSG:4326') if line_geoms else gpd.GeoDataFrame(columns=['corridor', 'geometry'], crs='EPSG:4326')

# 5) Plot 1: unemployment + project targets + crossing pressure
fig, ax = plt.subplots(figsize=(11, 10))
map_df.boundary.plot(ax=ax, color='#666', linewidth=0.5)
map_df.plot(ax=ax, column='unemployment_rate', cmap='YlOrRd', legend=True, legend_kwds={'label': 'Registered unemployment rate (%) - Dec 2025', 'shrink': 0.65}, missing_kwds={'color': '#f0f0f0'})
map_df[map_df['priority']].boundary.plot(ax=ax, color='#0b3d91', linewidth=2.0)

if not pts.empty:
    size = major_pts['persons_total'].fillna(0).astype(float) / 4000.0
    size = size.clip(lower=40, upper=400)
    major_pts.plot(ax=ax, markersize=size, color='#1f77b4', edgecolor='white', linewidth=0.6, alpha=0.85)

for _, r in map_df[map_df['priority']].iterrows():
    c = r.geometry.centroid
    ax.text(c.x, c.y, r['NAME_2'], fontsize=8, color='#111', ha='center', va='center')

for _, r in major_pts.iterrows():
    if pd.notna(r['persons_total']):
        label = f"{r['crossing_label']}\n{int(r['persons_total']):,} ppl/yr"
        ax.text(r.geometry.x + 0.04, r.geometry.y + 0.02, label, fontsize=7, color='#08306b')

ax.set_title('Lubelskie Target Areas: Unemployment Hotspots + Border Crossing Pressure (2025)', fontsize=12)
ax.set_axis_off()
plt.tight_layout()
fig.savefig(BASE / 'lubelskie_target_map_unemployment_border_2026-02-17.png', dpi=220)
plt.close(fig)

# 6) Plot 2: corridor concept map
fig, ax = plt.subplots(figsize=(11, 10))
map_df.plot(ax=ax, color='#f8f8f8', edgecolor='#777', linewidth=0.5)
map_df[map_df['priority']].plot(ax=ax, color='#fee8c8', edgecolor='#cc4c02', linewidth=1.2)

if not lines.empty:
    lines.plot(ax=ax, color='#3182bd', linewidth=2.0, alpha=0.8)
road_pts = pts[pts['crossing_label'].isin(['Dorohusk road', 'Zosin road', 'Dolhobyczow road', 'Hrebenne road'])].copy()
if not road_pts.empty:
    size = road_pts['trucks_total'].fillna(0).astype(float) / 600.0
    size = size.clip(lower=30, upper=260)
    road_pts.plot(ax=ax, markersize=size, color='#08519c', edgecolor='white', linewidth=0.6)

ax.scatter([lc.x], [lc.y], s=120, color='#31a354', edgecolor='white', zorder=5)
ax.text(lc.x + 0.05, lc.y, 'Lublin city hub', fontsize=8, color='#005a32')

for _, r in road_pts.iterrows():
    if pd.notna(r['trucks_total']):
        label = f"{r['crossing_label']}\n{int(r['trucks_total']):,} trucks/yr"
        ax.text(r.geometry.x + 0.03, r.geometry.y - 0.015, label, fontsize=7, color='#084594')

ax.set_title('Lubelskie Proposed Corridors: Lublin Hub to Border Gateways (Truck Flows, 2025)', fontsize=12)
ax.set_axis_off()
plt.tight_layout()
fig.savefig(BASE / 'lubelskie_target_map_corridors_2026-02-17.png', dpi=220)
plt.close(fig)

print('Generated files:')
for p in [
    BASE / 'lubelskie_unemployment_dec2025.csv',
    BASE / 'nosg_crossings_monthly_2025.csv',
    BASE / 'nosg_crossings_annual_2025.csv',
    BASE / 'lubelskie_target_map_unemployment_border_2026-02-17.png',
    BASE / 'lubelskie_target_map_corridors_2026-02-17.png',
]:
    print('-', p, p.stat().st_size)
