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
            
        # Optimization: Only consider the current node if it's not the target product
        is_target_product = root.product_info and root.product_info.get("Product", "").lower() == (target_name or "").lower()
        
        if not is_target_product:
            dist = self.euclidean_distance(target, root.point)
            c_neighbours.append((dist, root.product_info))
            c_neighbours.sort(key=lambda x: x[0])
            if len(c_neighbours) > k:
                c_neighbours.pop()
                
        axis = depth % self.k
        
        # Determine which branch to search first
        next_branch = root.left if target[axis] < root.point[axis] else root.right
        opposite_branch = root.right if target[axis] < root.point[axis] else root.left
        
        # Search the preferred branch
        self.knn_search(next_branch, target, k, depth + 1, c_neighbours, target_name)
        
        # Check if we need to search the opposite branch (sphere check)
        # We only search the opposite side if the hypersphere centered at the target
        # with radius equal to the distance to the farthest current neighbor
        # intersects the splitting hyperplane (defined by root.point[axis])
        if len(c_neighbours) < k: # If we haven't found k neighbors yet
            radius_sq = c_neighbours[-1][0] if c_neighbours else float('inf')
        else: # If we have k neighbors
            radius_sq = c_neighbours[-1][0]
        
        distance_to_hyperplane = abs(target[axis] - root.point[axis])
        
        # We search the opposite branch if the distance to the hyperplane is less than the current farthest neighbor distance
        if distance_to_hyperplane < radius_sq:
            self.knn_search(opposite_branch, target, k, depth + 1, c_neighbours, target_name)
            
        return c_neighbours

    def range_search(self, root, ranges_encoded, depth=0, results=None):
        if root is None:
            return []
        if results is None:
            results = []
            
        axis = depth % self.k
        
        # 1. Check if the current point is inside the range
        inside = True
        for i, (low_i, high_i) in enumerate(ranges_encoded):
            v = root.point[i]
            if v < low_i or v > high_i:
                inside = False
                break
        
        if inside:
            results.append(root.product_info)
            
        # 2. Recurse into children
        val_axis = root.point[axis]
        low_axis, high_axis = ranges_encoded[axis]

        # Go left if the lower boundary of the range is to the left of the splitting point
        if val_axis >= low_axis:
            self.range_search(root.left, ranges_encoded, depth + 1, results)
            
        # Go right if the upper boundary of the range is to the right of the splitting point
        if val_axis <= high_axis:
            self.range_search(root.right, ranges_encoded, depth + 1, results)
            
        return results

# ========================== Global variables (as App Context) ==========================
# Use relative path for the Flask app environment
DATA_CSV = r"C:\Users\D Praneeth\Downloads\SEM-3\PYTHON\Project\HM_TRIAL_4\cleaned_dataset.csv"
app = Flask(__name__)
app.secret_key = 'super_secret_key' # Needed for flash messages

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
median_defaults = {}
product_lookup = {}
kdtree = None

# ========================== Parsing and Encoding Functions (Modified for Flask) ==========================

def parse_price(x):
    try:
        if pd.isna(x) or x is None:
            return None
        return float(re.sub(r"[₹, ]+", "", str(x)))
    except:
        return None

def parse_ram_storage(s):
    if pd.isna(s) or s is None:
        return (None, None)
    s = str(s)
    nums = re.findall(r"(\d+)\s*GB", s, flags=re.IGNORECASE)
    if not nums:
        nums = re.findall(r"(\d+)\/", s)
        
    ram = int(nums[0]) if len(nums) >= 1 else None
    if len(nums) >= 2:
        storage = int(nums[1])
    elif len(nums) == 1:
        storage = None 
    else:
        storage = None
        
    return (ram, storage)

def parse_battery(s):
    if pd.isna(s) or s is None:
        return (None, None)
    s = str(s)
    m = re.search(r"(\d{3,6})\s*mAh", s, flags=re.IGNORECASE)
    w = re.search(r"(\d{1,3})\s*W", s, flags=re.IGNORECASE)
    batt = int(m.group(1)) if m else None
    watt = int(w.group(1)) if w else None
    return (batt, watt)

def parse_camera(s):
    if pd.isna(s) or s is None:
        return None
    s = str(s)
    nums = re.findall(r"(\d{1,4})\s*MP", s, flags=re.IGNORECASE)
    if nums:
        return int(nums[0]) 
    nums = re.findall(r"(\d{1,4})", s)
    return int(nums[0]) if nums else None

def parse_display(s):
    if pd.isna(s) or s is None:
        return None
    s = str(s)
    m = re.search(r"(\d+\.?\d*)\s*(?:\"|inch|inches|in)", s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    m = re.search(r"(\d+\.?\d*)", s)
    return float(m.group(1)) if m else None

def normalize_value(feat, val):
    lo, hi = scaler.get(feat, (0, 0))
    if val is None:
        val = median_defaults.get(feat, 0)
    try:
        val = float(val)
    except:
        val = float(median_defaults.get(feat, 0))
    
    if hi == lo:
        return 0.5 
        
    return (val - lo)/(hi - lo)

def encode_product(product):
    vec = []
    for feat in feature_list:
        val = product.get(feat)
        norm_val = normalize_value(feat, val)
        weighted_val = norm_val * weights.get(feat, 1.0)
        vec.append(weighted_val)
    return vec

def clamp01(x):
    if x is None: return 0.0
    try:
        if x<0: return 0.0
        if x>1: return 1.0
    except:
        return 0.0
    return x

def _clean_numeric_str(s):
    if s is None: return None
    return re.sub(r"[₹, ]+","",str(s))

def build_encoded_ranges(range_filters):
    encoded = []
    for feat in feature_list:
        lo_raw, hi_raw = range_filters.get(feat,(None,None))
        
        if feat == "Brand":
            continue
            
        lo_norm = 0.0 if lo_raw is None else normalize_value(feat, lo_raw)
        hi_norm = 1.0 if hi_raw is None else normalize_value(feat, hi_raw)
        
        if lo_norm>hi_norm: lo_norm,hi_norm = hi_norm,lo_norm
        
        w = weights.get(feat,1.0)
        encoded.append((clamp01(lo_norm)*w, clamp01(hi_norm)*w))
    return encoded


def compute_scaler_and_products():
    global scaler, median_defaults, product_lookup, kdtree
    
    try:
        df = pd.read_csv(DATA_CSV)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_CSV}' was not found.")
        return []

    df["Product"] = df.get("mobile_name", pd.Series(["Unknown"]*len(df))).fillna("Unknown")
    df["Brand"] = df["Product"].apply(lambda x: str(x).split()[0] if not pd.isna(x) else "Unknown")
    brand_map = {b:i+1 for i,b in enumerate(df["Brand"].unique())}
    
    df["Cost"] = df.get("price", pd.Series([None]*len(df))).apply(parse_price)
    
    avg_rating = df.get("avg_rating", pd.Series(dtype=float)).mean(skipna=True) if "avg_rating" in df.columns else 4.0
    df["Rating"] = df.get("avg_rating", pd.Series([None]*len(df))).fillna(avg_rating)
    
    parsed = []
    for _, row in df.iterrows():
        ram, storage = parse_ram_storage(row.get("ram_and_storage",""))
        batt, watt = parse_battery(row.get("battery_and_charging_speed",""))
        rear = parse_camera(row.get("rear_camera",""))
        front = parse_camera(row.get("front_camera",""))
        disp = parse_display(row.get("display",""))
        be = brand_map.get(row.get("Brand"),0)
        
        product_data = {
            "Cost": row.get("Cost"), "Rating": row.get("Rating"),
            "Brand_encoded": be, "RAM": ram, "Storage": storage,
            "Battery_mAh": batt, "Charging_W": watt, "Rear_MP": rear,
            "Front_MP": front, "Display_in": disp, "Product": row.get("Product"),
            "Brand": row.get("Brand"),
        }
        parsed.append(product_data)

    scaler.clear()
    for feat in feature_list:
        vals = [p[feat] for p in parsed if p[feat] is not None]
        min_val = min(vals) if vals else 0
        max_val = max(vals) if vals else 0
        if min_val == max_val:
            scaler[feat] = (min_val, max_val + 0.001)
        else:
            scaler[feat] = (min_val, max_val)

    median_defaults.clear()
    for feat in feature_list:
        vals = [p[feat] for p in parsed if p[feat] is not None]
        median_defaults[feat] = statistics.median(vals) if vals else 0

    products_for_tree = []
    product_lookup.clear()
    for p in parsed:
        vec = encode_product(p)
        products_for_tree.append((vec, p))
        product_lookup[p["Product"].lower()] = p

    kdtree = KDTree(k=len(feature_list))
    kdtree.build_tree(products_for_tree)
    
    print(f"Data loaded, Scaler/Median computed, and KD-Tree built with {len(products_for_tree)} products.")
    return products_for_tree


# ========================== Flask Routes ==========================

@app.before_request
def setup_data():
    if kdtree is None:
        compute_scaler_and_products()

# NEW: API endpoint for live search suggestions
@app.route('/api/search_phones')
def search_phones():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    # Find matching product names
    matches = [
        product_lookup[key]['Product'] for key in product_lookup 
        if query in key
    ]
    
    return jsonify(matches[:10]) # Return top 10 matches

@app.route('/')
def index():
    # --- Get search, filter, and pagination parameters from URL ---
    phone_name_query = request.args.get('phone_name', '').strip()
    page = request.args.get('page', 1, type=int)
    
    # If no phone name is provided, just render the template
    if not phone_name_query:
        return render_template('index.html', feature_list=feature_list, phone_name_query="")

    # --- Find the selected phone ---
    selected_phone = product_lookup.get(phone_name_query.lower())
    if not selected_phone:
        flash(f"Phone '{phone_name_query}' not found. Please select a valid phone from the suggestions.", 'danger')
        return render_template('index.html', feature_list=feature_list, phone_name_query=phone_name_query)

    # --- Process Filters ---
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
    
    # Check if any filters are active
    use_filters = len(raw_filters) > 0
    
    vector = encode_product(selected_phone)
    full_results = []
    
    # --- Perform Search (KNN or Filtered) ---
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
            # Linear search among filtered results
            neighs = []
            for p in final_candidates:
                vec2 = encode_product(p)
                dist = math.sqrt(sum((vec2[i]-vector[i])**2 for i in range(len(vector))))
                neighs.append((dist, p))
            
            neighs.sort(key=lambda x: x[0])
            full_results = neighs
            
    else: # No filters, use standard KNN
        # We search for a large number of neighbors to fill pages
        # The number of products is ~1200, so searching for 1500 is safe
        knn = kdtree.knn_search(kdtree.root, vector, 1500, target_name=phone_name_query)
        uniq = []; seen = set()
        for dist, info in knn:
            name = info.get("Product", "").lower()
            if name not in seen and name != phone_name_query.lower():
                uniq.append((dist, info))
                seen.add(name)
        full_results = uniq

    # --- Pagination Logic ---
    PER_PAGE = 20
    total_results = len(full_results)
    total_pages = math.ceil(total_results / PER_PAGE)
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    paginated_results = full_results[start:end]

    # Preserve query args for pagination links
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