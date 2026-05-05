import pandas as pd
import re
import math
import statistics
import sys
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from urllib.parse import urlencode

# ========================== KD-Tree Classes (Unchanged) ==========================
class KDNode:
    def __init__(self, point=None, product_info=None, left=None, right=None):
        self.point = point
        self.product_info = product_info
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, k):
        self.k = k
        self.root = None
    
    def build_tree(self, points, depth=0):
        if not points:
            return None
        axis = depth % self.k
        points.sort(key=lambda x: x[0][axis])
        median_idx = len(points) // 2
        node = KDNode(
            point=points[median_idx][0],
            product_info=points[median_idx][1],
            left=self.build_tree(points[:median_idx], depth + 1),
            right=self.build_tree(points[median_idx + 1:], depth + 1)
        )
        if depth == 0:
            self.root = node
        return node

    def euclidean_distance(self, p1, p2):
        return math.sqrt(sum((p1[i]-p2[i])**2 for i in range(len(p1))))

    def knn_search(self, root, target, k, depth=0, c_neighbours=None, target_name=None):
        if root is None:
            return []
        if c_neighbours is None:
            c_neighbours = []
            
        is_target_product = root.product_info and root.product_info.get("Product", "").lower() == (target_name or "").lower()
        
        if not is_target_product:
            dist = self.euclidean_distance(target, root.point)
            c_neighbours.append((dist, root.product_info))
            c_neighbours.sort(key=lambda x: x[0])
            if len(c_neighbours) > k:
                c_neighbours.pop()
                
        axis = depth % self.k
        
        next_branch = root.left if target[axis] < root.point[axis] else root.right
        opposite_branch = root.right if target[axis] < root.point[axis] else root.left
        
        self.knn_search(next_branch, target, k, depth + 1, c_neighbours, target_name)
        
        # radius is distance to farthest neighbour in list (if any)
        radius = c_neighbours[-1][0] if c_neighbours else float('inf')
        distance_to_hyperplane = abs(target[axis] - root.point[axis])
        
        if distance_to_hyperplane < radius:
            self.knn_search(opposite_branch, target, k, depth + 1, c_neighbours, target_name)
            
        return c_neighbours

    def range_search(self, root, ranges_encoded, depth=0, results=None):
        if root is None:
            return []
        if results is None:
            results = []
            
        axis = depth % self.k
        
        inside = True
        for i, (low_i, high_i) in enumerate(ranges_encoded):
            v = root.point[i]
            if v < low_i or v > high_i:
                inside = False
                break
        
        if inside:
            results.append(root.product_info)
            
        val_axis = root.point[axis]
        low_axis, high_axis = ranges_encoded[axis]

        if val_axis >= low_axis:
            self.range_search(root.left, ranges_encoded, depth + 1, results)
        if val_axis <= high_axis:
            self.range_search(root.right, ranges_encoded, depth + 1, results)
            
        return results

# ========================== Global variables (as App Context) ==========================
DATA_CSV = r"C:\Users\D Praneeth\Downloads\python 3rd sem\Project\HM_TRIAL_4\cleaned_dataset.csv"
app = Flask(__name__)
app.secret_key = 'super_secret_key'  # keep using env var in production

feature_list = [
    "Cost", "Rating", "Brand_encoded", "RAM", "Storage", "Battery_mAh",
    "Charging_W", "Rear_MP", "Front_MP", "Display_in"
]

weights = {
    "Cost": 1.0, "Rating": 1.0, "Brand_encoded": 1.0, "RAM": 1.0, "Storage": 1.0,
    "Battery_mAh": 1.0, "Charging_W": 1.0, "Rear_MP": 1.0, "Front_MP": 1.0,
    "Display_in": 1.0
}

# These will be populated once
scaler = {}
product_lookup = {}
kdtree = None

def parse_spec(s, spec_type):
    # simplified: dataset assumed to be present and well-formed
    if s is None:
        return None
    s = str(s).strip()

    if spec_type == "ram_storage":
        m = re.findall(r"(\d+)\s*GB", s, flags=re.IGNORECASE)
        if not m:
            m = re.findall(r"(\d+)\/", s)
        ram = int(m[0]) if len(m) >= 1 else None
        storage = int(m[1]) if len(m) >= 2 else None
        return (ram, storage)

    elif spec_type == "battery":
        m_mah = re.search(r"(\d{3,6})\s*mAh", s, flags=re.IGNORECASE)
        m_watt = re.search(r"(\d{1,3})\s*W", s, flags=re.IGNORECASE)
        batt = int(m_mah.group(1)) if m_mah else None
        watt = int(m_watt.group(1)) if m_watt else None
        return (batt, watt)

    elif spec_type == "camera":
        m = re.search(r"(\d{1,4})\s*MP", s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m_alt = re.search(r"(\d{1,4})", s)
        return int(m_alt.group(1)) if m_alt else None

    elif spec_type == "display":
        m = re.search(r"(\d+\.?\d*)\s*(\"|inch|inches|in)", s, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        m_alt = re.search(r"(\d+\.?\d*)", s)
        return float(m_alt.group(1)) if m_alt else None

    elif spec_type == "price":
        m = re.sub(r"[₹, ]+", "", s)
        try:
            return float(m)
        except:
            return None

    else:
        return None

def compute_scaler_and_products():
    global scaler, product_lookup, kdtree

    try:
        df = pd.read_csv(DATA_CSV)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_CSV}' was not found.")
        return []

    # assume dataset has full values
    df["Product"] = df.get("mobile_name", pd.Series(["Unknown"] * len(df)))
    df["Brand"] = df["Product"].apply(lambda x: str(x).split()[0])
    brand_map = {b: i + 1 for i, b in enumerate(df["Brand"].unique())}

    # Use parse_spec for price
    df["Cost"] = df.get("price", pd.Series([None] * len(df))).apply(lambda x: parse_spec(x, "price"))

    avg_rating = df.get("avg_rating", pd.Series(dtype=float)).mean(skipna=True) if "avg_rating" in df.columns else 4.0
    df["Rating"] = df.get("avg_rating", pd.Series([avg_rating] * len(df)))

    parsed = []
    for _, row in df.iterrows():
        ram, storage = parse_spec(row.get("ram_and_storage", ""), "ram_storage")
        batt, watt = parse_spec(row.get("battery_and_charging_speed", ""), "battery")
        rear = parse_spec(row.get("rear_camera", ""), "camera")
        front = parse_spec(row.get("front_camera", ""), "camera")
        disp = parse_spec(row.get("display", ""), "display")
        be = brand_map.get(row.get("Brand"), 0)

        product_data = {
            "Cost": row.get("Cost"),
            "Rating": row.get("Rating"),
            "Brand_encoded": be,
            "RAM": ram,
            "Storage": storage,
            "Battery_mAh": batt,
            "Charging_W": watt,
            "Rear_MP": rear,
            "Front_MP": front,
            "Display_in": disp,
            "Product": row.get("Product"),
            "Brand": row.get("Brand"),
        }
        parsed.append(product_data)

    # Compute scaler (min, max) for each feature
    scaler.clear()
    for feat in feature_list:
        vals = [p[feat] for p in parsed if p[feat] is not None]
        min_val = min(vals) if vals else 0
        max_val = max(vals) if vals else 0
        # add tiny epsilon when min==max to avoid division by zero
        scaler[feat] = (min_val, max_val + 1e-6) if min_val == max_val else (min_val, max_val)

    # Build KD-Tree
    products_for_tree = []
    product_lookup.clear()
    for p in parsed:
        vec = encode_product(p)
        products_for_tree.append((vec, p))
        product_lookup[p["Product"].lower()] = p

    kdtree = KDTree(k=len(feature_list))
    kdtree.build_tree(products_for_tree)

    print(f"Data loaded, Scaler computed, and KD-Tree built with {len(products_for_tree)} products.")
    return products_for_tree

def normalize_value(feat, val):
    lo, hi = scaler.get(feat, (0, 1))
    try:
        val = float(val)
    except:
        val = float(lo)  
    return (val - lo) / (hi - lo)

def encode_product(product):
    vec = []
    for feat in feature_list:
        val = product.get(feat)
        norm_val = normalize_value(feat, val)
        weighted_val = norm_val * weights.get(feat, 1.0)
        # clamp to [0,1] just in case
        if weighted_val < 0: weighted_val = 0.0
        if weighted_val > 1: weighted_val = 1.0
        vec.append(weighted_val)
    return vec

def _clean_numeric_str(s):
    if s is None: return None
    return re.sub(r"[₹, ]+","",str(s))

def build_encoded_ranges(range_filters):
    encoded = []
    for feat in feature_list:
        lo_raw, hi_raw = range_filters.get(feat,(None,None))
        lo_norm = 0.0 if lo_raw is None else normalize_value(feat, lo_raw)
        hi_norm = 1.0 if hi_raw is None else normalize_value(feat, hi_raw)
        if lo_norm > hi_norm: lo_norm, hi_norm = hi_norm, lo_norm
        w = weights.get(feat,1.0)
        encoded.append((max(0.0, min(1.0, lo_norm))*w, max(0.0, min(1.0, hi_norm))*w))
    return encoded

# ========================== Flask Routes ==========================
@app.before_request
def setup_data():
    if kdtree is None:
        compute_scaler_and_products()

@app.route('/api/search_phones')
def search_phones():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    matches = [product_lookup[key]['Product'] for key in product_lookup if query in key]
    return jsonify(matches[:10])

@app.route('/')
def index():
    phone_name_query = request.args.get('phone_name', '').strip()
    page = request.args.get('page', 1, type=int)
    if not phone_name_query:
        return render_template('index.html', feature_list=feature_list, phone_name_query="")

    selected_phone = product_lookup.get(phone_name_query.lower())
    if not selected_phone:
        flash(f"Phone '{phone_name_query}' not found. Please select a valid phone from the suggestions.", 'danger')
        return render_template('index.html', feature_list=feature_list, phone_name_query=phone_name_query)

    raw_filters = {}
    brand_filter_input = request.args.get('brand_filter', '').strip()
    if brand_filter_input:
        raw_filters['Brand'] = [v.strip().lower() for v in brand_filter_input.split(",") if v.strip()]

    for feat in feature_list:
        if feat == "Brand_encoded": continue
        min_val = _clean_numeric_str(request.args.get(f'{feat.lower()}_min'))
        max_val = _clean_numeric_str(request.args.get(f'{feat.lower()}_max'))
        if min_val or max_val:
             try:
                min_f = float(min_val) if min_val else None
                max_f = float(max_val) if max_val else None
                raw_filters[feat] = (min_f, max_f)
             except (ValueError, TypeError):
                pass

    use_filters = len(raw_filters) > 0
    vector = encode_product(selected_phone)
    full_results = []

    if use_filters:
        encoded_ranges = build_encoded_ranges(raw_filters)
        filtered_candidates = kdtree.range_search(kdtree.root, encoded_ranges)
        brand_filter = raw_filters.get("Brand")
        final_candidates = [
            p for p in filtered_candidates
            if p["Product"].lower() != phone_name_query.lower() and
               (brand_filter is None or p["Brand"].lower() in brand_filter)
        ]
        if not final_candidates:
            flash("No products match the selected filters. Please relax your criteria.", 'info')
        else:
            neighs = []
            for p in final_candidates:
                vec2 = encode_product(p)
                dist = math.sqrt(sum((vec2[i]-vector[i])**2 for i in range(len(vector))))
                neighs.append((dist, p))
            neighs.sort(key=lambda x: x[0])
            full_results = neighs
    else:
        knn = kdtree.knn_search(kdtree.root, vector, 1500, target_name=phone_name_query)
        uniq = []; seen = set()
        for dist, info in knn:
            name = info.get("Product", "").lower()
            if name not in seen and name != phone_name_query.lower():
                uniq.append((dist, info))
                seen.add(name)
        full_results = uniq

    PER_PAGE = 20
    total_results = len(full_results)
    total_pages = math.ceil(total_results / PER_PAGE)
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    paginated_results = full_results[start:end]

    query_params = request.args.to_dict()
    return render_template('index.html', 
                           selected_phone=selected_phone, 
                           recommendations=paginated_results,
                           feature_list=feature_list,
                           phone_name_query=selected_phone.get('Product'),
                           page=page,
                           total_pages=total_pages,
                           query_params=query_params,
                           urlencode=urlencode)

if __name__ == '__main__':
    t0 = time.time()
    compute_scaler_and_products()
    t1 = time.time()
    print(f"Flask App Ready. KD-Tree setup time: {t1-t0:.4f} seconds.")
    app.run(debug=True)
