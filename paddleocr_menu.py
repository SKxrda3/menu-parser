# from paddleocr import PaddleOCR
# import os
# import re
# import pandas as pd
# from tabulate import tabulate
# import mysql.connector as mysql


# # ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# def extract_boxes(image_path, conf_threshold=0.6):
#     results = ocr.predict(image_path)
#     boxes = []

#     # results is a list with one dict per image/page
#     result = results[0]  # take first element

#     dt_polys = result['dt_polys']       # list of arrays of box points
#     rec_texts = result['rec_texts']     # list of text strings
#     rec_scores = result['rec_scores']   # list of confidences

#     for box_points, text, conf in zip(dt_polys, rec_texts, rec_scores):
#         if conf >= conf_threshold:
#             # box_points is a numpy array with shape (4,2) or similar
#             box = box_points.tolist()  # convert to list of points if you want
#             x_center = sum([p[0] for p in box]) / len(box)
#             y_center = sum([p[1] for p in box]) / len(box)
#             boxes.append({
#                 'text': text.strip(),
#                 'conf': conf,
#                 'box': box,
#                 'x': x_center,
#                 'y': y_center
#             })
#     return boxes



# def group_by_rows(boxes, y_thresh=15):
#     boxes.sort(key=lambda b: b["y"])
#     rows = []
#     current_row = []
#     last_y = -1000

#     for box in boxes:
#         if abs(box["y"] - last_y) > y_thresh:
#             if current_row:
#                 rows.append(sorted(current_row, key=lambda b: b["x"]))  # Sort left-to-right
#             current_row = [box]
#         else:
#             current_row.append(box)
#         last_y = box["y"]

#     if current_row:
#         rows.append(sorted(current_row, key=lambda b: b["x"]))
#     return rows


# def assign_categories(rows):
#     categorized_rows = []
#     current_category = "Uncategorized"

#     for row in rows:
#         texts = [box["text"] for box in row]
#         full_line = " ".join(texts).strip()

#         # Skip row if it has a price — likely an item line
#         if detect_price(full_line):
#             for box in row:
#                 box["category"] = current_category
#             categorized_rows.append(row)
#             continue

#         # Better category detection
#         words = full_line.split()
#         uppercase_words = sum(1 for w in words if w.isupper() or w.istitle())
#         is_probable_category = (
#             uppercase_words >= max(1, len(words) // 2) and
#             len(full_line) <= 35 and
#             len(words) <= 4
#         )

#         if is_probable_category:
#             current_category = full_line
#             continue  # Skip adding as data row

#         for box in row:
#             box["category"] = current_category
#         categorized_rows.append(row)

#     return categorized_rows


# def detect_price(text):
#     return re.search(r'(\u20B9|Rs\.?)?\s?\d{1,4}([.,]\d{1,2})?', text)

# def is_valid_item(text):
#     if not text or len(text.strip()) <= 2:
#         return False

#     if text.strip().isupper() and len(text.strip()) <= 3:
#         return False

#     noise_keywords = {
#         'am', 'pm', 'AM', 'PM', 'yo', 'l', 't', 'a', 'b',
#         '/', '-', '|', ':', '.', ',', '–', '—', '_', '(', ')',
#         'AM to', 'PM to', 'TO', 'to', 'and', '&', '*',
#         'daily', 'only', 'each', 'per', 'day', 'week', 'month',
#         'with', 'served', 'served with', 'includes', 'including',
#         'plate', 'pcs', 'pieces', 'item', 'items', 'available',
#         'fri', 'mon', 'tue', 'wed', 'thu', 'sat', 'sun',
#         'mon-fri', 'sat-sun', 'mon-thu', 'fri-sun',
#         'timings', 'timing', 'from', 'at', 'till', 'until',
#         'for', 'special', 'offer', 'limited', 'extra', 'add-on',
#         'optional', 'choose', 'selection', 'assorted',
#         'combo', 'set', 'option', 'mrp', 'gst', 'inclusive',
#         'exclusive', 'taxes', 'inc.', 'excl.', 'incl.'
#     }

#     clean_text = re.sub(r'[^\w]', '', text).lower()
#     if clean_text in noise_keywords:
#         return False

#     if not re.search(r'[a-zA-Z]', text):
#         return False

#     return True

# def parse_rows_to_menu(categorized_rows, image_name="unknown"):
#     menu = []
#     last_item_entry = None

#     for row in categorized_rows:
#         row.sort(key=lambda b: b["x"])
#         full_line = " ".join([b["text"] for b in row]).strip()
#         current_category = row[0].get("category", "Uncategorized")

#         # Search for all prices in the row
#         price_matches = list(re.finditer(r'(₹|Rs\.?)?\s?\d{1,5}([.,]\d{1,2})?', full_line))

#         if price_matches:
#             items = []
#             prices = []
#             start = 0

#             for idx, match in enumerate(price_matches):
#                 price_text = match.group().strip()
#                 price_start = match.start()

#                 # Get text before this price as the item
#                 item_chunk = full_line[start:price_start].strip(" -–—|,")
#                 item_texts = re.split(r'\s{2,}|,|/| - | \| |\. ', item_chunk)

#                 for item_text in item_texts:
#                     # item_text = item_text.strip()
#                     item_text = re.sub(r'\(.*?\)', '', item_text).strip() 

#                     if not is_valid_item(item_text):
#                         continue

#                     items.append(item_text)
#                     prices.append(price_text)

#                 start = match.end()

#             for i in range(min(len(items), len(prices))):
#                 entry = {
#                     "image": image_name,
#                     "category": current_category,
#                     "item": items[i],
#                     "price": prices[i],
#                     "description": ""
#                 }
#                 last_item_entry = entry
#                 menu.append(entry)

#         else:
#             if last_item_entry and current_category == last_item_entry["category"]:
#                 # Add as continuation to description
#                 last_item_entry["description"] = (last_item_entry.get("description", "") + " " + full_line).strip()

#     return menu

# def print_menu_table(menu):
#     headers = ["Item", "Category", "Price"]
#     rows = [[entry["item"], entry["category"], entry["price"]] for entry in menu]
#     print(tabulate(rows, headers=headers, tablefmt="grid"))

# def process_folder(folder_path, conf_threshold=0.6, y_thresh=15, output_excel="output1.xlsx"):
#     all_data = []

#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith((".jpg", ".jpeg", ".png",".bmp", ".tiff")):
#             image_path = os.path.join(folder_path, filename)
#             print(f"Processing {filename} ...")

#             boxes = extract_boxes(image_path, conf_threshold=conf_threshold)
#             rows = group_by_rows(boxes, y_thresh=y_thresh)
#             final_data = assign_categories(rows)

#             df = pd.DataFrame(final_data)
#             print(tabulate(df, headers="keys"))

#             menu = parse_rows_to_menu(rows)
#             ignore_phrases = [
#                 "preparation time",
#                 "serving size",
#                 "cooking time",
#                 "calories",
#                 # aur aise words add kar sakte ho
#             ]
#             for entry in menu:
#                 combined_text = (entry["item"] + " " + entry["category"] + " " + entry.get("description", "")).lower()

#                 if any(phrase in combined_text for phrase in ignore_phrases):
#                     continue  # skip this entry
                        
#                 all_data.append({
#                     "Image": filename,
#                     "Category": entry["category"],
#                     "Item": entry["item"],
#                     "Price": entry["price"],
#                     "Description": entry.get("description", "")
#                 })

#     if not all_data:
#         print("No menu data extracted from images.")
#         return

#     # df = pd.DataFrame(all_data, columns=["Image", "Category", "Item", "Price","Description"])
#     # df.to_excel(output_excel, index=False)
#     # print(f"\n\u2705 Extracted menu data saved to {output_excel}")

# # Initialize PaddleOCR (adjust lang as needed)
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Path to the folder containing your menu images
# image_folder = "path_to_your_menu_images"

# # MySQL database credentials
# MYSQL_HOST = "localhost"
# MYSQL_USER = "your_mysql_username"
# MYSQL_PASSWORD = "your_mysql_password"
# MYSQL_DATABASE = "webinfinity_sb"

# # Loop through each image and run OCR
# all_data = []
# for filename in os.listdir(image_folder):
#     if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         image_path = os.path.join(image_folder, filename)
#         result = ocr.ocr(image_path, cls=True)

#         # Process each line detected by OCR
#         for line in result[0]:
#             text = line[1][0]

#             # Basic logic to split item and price
#             if "$" in text:
#                 parts = text.rsplit("$", 1)
#                 item = parts[0].strip()
#                 price = "$" + parts[1].strip()
#                 all_data.append({
#                     "Image": filename,
#                     "Category": "",  # Optional: you can add category logic here
#                     "Item": item,
#                     "Price": price,
#                     "Description": ""  # Optional: extract description if needed
#                 })

#     # ✅ Insert data into MySQL
#     try:
#         connection = mysql.connect(
#             host=MYSQL_HOST,
#             user=MYSQL_USER,
#             password=MYSQL_PASSWORD,
#             database=MYSQL_DATABASE
#         )
#         cursor = connection.cursor()

#         insert_query = """
#         INSERT INTO menu_or_services (image, category, item, price, description)
#         VALUES (%s, %s, %s, %s, %s)
#         """

#         for entry in all_data:
#             cursor.execute(insert_query, (
#                 entry["Image"],
#                 entry["Category"],
#                 entry["Item"],
#                 entry["Price"],
#                 entry["Description"]
#             ))

#         connection.commit()
#         print(f"\n✅ Extracted menu data inserted into MySQL database.")
#     except mysql.Error as err:
#         print(f"❌ MySQL error: {err}")
#     finally:
#         if connection.is_connected():
#             cursor.close()
#             connection.close()


#     print("\n\ud83d\udccb Combined Extracted Menu:")
#     print(tabulate(df[["Item", "Category", "Price"]], headers=["Item", "Category", "Price"], tablefmt="grid"))

# if __name__ == "__main__":
#     folder_path = "menu"  # Change this to your folder path
#     process_folder(folder_path)


from paddleocr import PaddleOCR
import os
import re
import pandas as pd
from tabulate import tabulate
import mysql.connector as mysql
import os
from dotenv import load_dotenv

load_dotenv()

mysql_config = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}


# ocr = PaddleOCR(use_angle_cls=True, lang='en')
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    det_model_dir='en_PP-OCRv3_det_infer',
    rec_model_dir='en_PP-OCRv3_rec_infer',
    use_gpu=False
)

def extract_boxes(image_path, conf_threshold=0.6):
    results = ocr.predict(image_path)
    boxes = []
    result = results[0]

    for box_points, text, conf in zip(result['dt_polys'], result['rec_texts'], result['rec_scores']):
        if conf >= conf_threshold:
            box = box_points.tolist()
            x_center = sum([p[0] for p in box]) / len(box)
            y_center = sum([p[1] for p in box]) / len(box)
            boxes.append({
                'text': text.strip(),
                'conf': conf,
                'box': box,
                'x': x_center,
                'y': y_center
            })
    return boxes


def group_by_rows(boxes, y_thresh=15):
    boxes.sort(key=lambda b: b["y"])
    rows = []
    current_row = []
    last_y = -1000

    for box in boxes:
        if abs(box["y"] - last_y) > y_thresh:
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b["x"]))
            current_row = [box]
        else:
            current_row.append(box)
        last_y = box["y"]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b["x"]))
    return rows


def assign_categories(rows):
    categorized_rows = []
    current_category = "Uncategorized"

    for row in rows:
        texts = [box["text"] for box in row]
        full_line = " ".join(texts).strip()

        if detect_price(full_line):
            for box in row:
                box["category"] = current_category
            categorized_rows.append(row)
            continue

        words = full_line.split()
        uppercase_words = sum(1 for w in words if w.isupper() or w.istitle())
        is_probable_category = (
            uppercase_words >= max(1, len(words) // 2) and
            len(full_line) <= 35 and
            len(words) <= 4
        )

        if is_probable_category:
            current_category = full_line
            continue

        for box in row:
            box["category"] = current_category
        categorized_rows.append(row)

    return categorized_rows


def detect_price(text):
    return re.search(r'(\u20B9|Rs\.?)?\s?\d{1,4}([.,]\d{1,2})?', text)


def is_valid_item(text):
    if not text or len(text.strip()) <= 2:
        return False

    if text.strip().isupper() and len(text.strip()) <= 3:
        return False

    noise_keywords = { 'am', 'pm', 'yo', 'l', 't', 'a', 'b', '/', '-', '|', ':', '.', ',', '–', '—', '_', '(', ')',
                       'daily', 'only', 'each', 'per', 'day', 'week', 'month', 'with', 'served', 'includes',
                       'available', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'timings', 'timing',
                       'from', 'at', 'till', 'until', 'for', 'special', 'offer', 'extra', 'add-on',
                       'optional', 'combo', 'set', 'option', 'mrp', 'gst', 'inclusive', 'exclusive',
                       'taxes', 'inc.', 'excl.', 'incl.' }

    clean_text = re.sub(r'[^\w]', '', text).lower()
    if clean_text in noise_keywords:
        return False

    if not re.search(r'[a-zA-Z]', text):
        return False

    return True


def parse_rows_to_menu(categorized_rows, image_name="unknown"):
    menu = []
    last_item_entry = None

    for row in categorized_rows:
        row.sort(key=lambda b: b["x"])
        full_line = " ".join([b["text"] for b in row]).strip()
        current_category = row[0].get("category", "Uncategorized")

        price_matches = list(re.finditer(r'(₹|Rs\.?)?\s?\d{1,5}([.,]\d{1,2})?', full_line))

        if price_matches:
            items = []
            prices = []
            start = 0

            for match in price_matches:
                price_text = match.group().strip()
                price_start = match.start()
                item_chunk = full_line[start:price_start].strip(" -–—|,")
                item_texts = re.split(r'\s{2,}|,|/| - | \| |\. ', item_chunk)

                for item_text in item_texts:
                    item_text = re.sub(r'\(.*?\)', '', item_text).strip()
                    if not is_valid_item(item_text):
                        continue
                    items.append(item_text)
                    prices.append(price_text)

                start = match.end()

            for i in range(min(len(items), len(prices))):
                entry = {
                    "image": image_name,
                    "category": current_category,
                    "item": items[i],
                    "price": prices[i],
                    "description": ""
                }
                last_item_entry = entry
                menu.append(entry)

        elif last_item_entry and current_category == last_item_entry["category"]:
            last_item_entry["description"] += " " + full_line

    return menu


# def insert_into_mysql(data, host, user, password, database):
def insert_into_mysql(data, host, user, password, database, vender_id):
    connection = None
    cursor = None
    try:
        connection = mysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=3306,
            use_pure=True
        )
        cursor = connection.cursor()

        insert_query = """
        INSERT INTO menu_or_services (category, item_or_service, price, description, vendor_id, image_path)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        for entry in data:
            cursor.execute(insert_query, (
                entry["category"],
                entry["item"],
                entry["price"],
                entry["description"],
                vender_id,
                entry["image"]  # Assuming you're storing the filename
            ))


        connection.commit()
        print("\n✅ Extracted menu data inserted into MySQL database.")

    except mysql.Error as err:
        print(f"❌ MySQL error: {err}")

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()



# def process_folder(folder_path, mysql_config):
def process_folder(folder_path, mysql_config, vender_id):

    all_data = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename} ...")

            boxes = extract_boxes(image_path)
            rows = group_by_rows(boxes)
            final_data = assign_categories(rows)
            menu = parse_rows_to_menu(final_data, image_name=filename)

            ignore_phrases = ["preparation time", "serving size", "cooking time", "calories"]

            for entry in menu:
                combined_text = (entry["item"] + " " + entry["category"] + " " + entry.get("description", "")).lower()
                if any(phrase in combined_text for phrase in ignore_phrases):
                    continue
                all_data.append(entry)

    if all_data:
        # insert_into_mysql(all_data, **mysql_config)
        insert_into_mysql(all_data, vender_id=vender_id, **mysql_config)
    else:
        print("No menu data extracted from images.")


if __name__ == "__main__":
    folder_path = "menu1"  # Your image folder path
    vender_id = 1         # Replace with actual vender_id
    process_folder(folder_path, mysql_config, vender_id)
    # mysql_config = {
    #     "host": "localhost",
    #     "user": "your_mysql_username",
    #     "password": "your_mysql_password",
    #     "database": "webinfinity_sb"
    # }
    # process_folder(folder_path, mysql_config)
