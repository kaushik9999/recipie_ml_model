import pandas as pd
import yaml
import ast # Import the ast module
from sklearn.model_selection import train_test_split
import os # Import os for path joining and existence check

# Load parameters from params.yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Define file paths
RAW_DATA_PATH = './data/raw/recipes_data.csv' # This is the dataset with recipes
PROCESSED_DATA_DIR = './data/processed/'
TRAIN_DATA_PATH = f'{PROCESSED_DATA_DIR}/train.csv'
TEST_DATA_PATH = f'{PROCESSED_DATA_DIR}/test.csv'

print("Loading raw data...")

df = pd.DataFrame() # Initialize an empty DataFrame
initial_line_count = 0
skipped_lines_count = 0

try:
    # Get total number of lines in the raw data file
    if os.path.exists(RAW_DATA_PATH):
        with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
            # Read header line first if exists, then count the rest
            try:
                next(f) # Skip header line for counting data rows
                initial_line_count = sum(1 for line in f) # Count data rows
                initial_line_count += 1 # Add header line back to total count
            except StopIteration:
                 # Handle empty file case
                 initial_line_count = 0
            print(f"Total lines in raw data file (including header): {initial_line_count}")

    # First attempt - try tab separator
    df = pd.read_csv(RAW_DATA_PATH, sep='\t', encoding='utf-8', on_bad_lines='skip', engine='python')
    
    # Check if all columns are in one column (indicates wrong separator)
    if len(df.columns) == 1 and ',' in df.columns[0]:
        print("Data appears to be comma-separated rather than tab-separated. Reloading with comma separator...")
        column_names = df.columns[0].split(',')
        
        # Reload with comma separator
        df = pd.read_csv(RAW_DATA_PATH, sep=',', encoding='utf-8', on_bad_lines='skip', engine='python', 
                         names=column_names, header=0)
    
    # Calculate skipped lines
    if initial_line_count > 0:
        rows_read_successfully = df.shape[0]
        skipped_lines_count = (initial_line_count - 1) - rows_read_successfully

    print("Data loaded successfully, skipping bad lines.")
    print(f"Successfully loaded {df.shape[0]} data rows.")
    print(f"Ignored lines during loading: {skipped_lines_count}")
    print("Columns:", df.columns.tolist())

except FileNotFoundError:
    print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
    print("Please make sure your raw data is in the data/raw/ directory and update RAW_DATA_PATH in this script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    print("Could not complete data loading due to an unexpected error.")
    exit()

print("Starting data preprocessing...")

# --- Data Cleaning and Feature Engineering Section ---
# Handle missing values
initial_rows_before_dropna = df.shape[0]
df.dropna(inplace=True)
rows_after_dropna = df.shape[0]
dropped_by_dropna_count = initial_rows_before_dropna - rows_after_dropna
print(f"Dropped {dropped_by_dropna_count} rows with missing values.")

# Parse string representations of lists for ingredients, directions, and NER
columns_to_parse = ['ingredients', 'directions', 'NER']

# First, ensure the column names are properly matched (case-insensitive)
column_mapping = {}
for col in df.columns:
    for target_col in columns_to_parse:
        if col.lower() == target_col.lower():
            column_mapping[target_col] = col

print(f"Column mapping: {column_mapping}")

# For title and other string columns
title_col = None
link_col = None
site_col = None

for col in df.columns:
    if col.lower() == 'title':
        title_col = col
    elif col.lower() == 'link':
        link_col = col
    elif col.lower() == 'site':
        site_col = col

print(f"Title column: {title_col}")
print(f"Link column: {link_col}")
print(f"Site column: {site_col}")

# Now parse the columns using the mapping
for target_col, actual_col in column_mapping.items():
    try:
        print(f"Parsing column {actual_col}, sample value: {df[actual_col].iloc[0]}")
        df[actual_col] = df[actual_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        print(f"Successfully parsed column '{actual_col}'.")
    except Exception as e:
        print(f"Warning: Could not parse column '{actual_col}'. Error: {e}")
        # Instead of dropping rows, we'll try to fix common issues
        # If the list is malformed, attempt to clean it up
        try:
            df[actual_col] = df[actual_col].apply(
                lambda x: ast.literal_eval(x.replace('""', '"').strip()) 
                if isinstance(x, str) else x
            )
            print(f"Attempted to fix and parse column '{actual_col}'.")
        except Exception as e2:
            print(f"Could not fix parsing issues in '{actual_col}'. Error: {e2}")
            print(f"Sample value that caused error: {df[actual_col].iloc[0]}")

# If the columns weren't properly parsed, we'll need to adapt
if not column_mapping or len(column_mapping) < 2:
    print("Warning: Could not identify proper columns. Attempting direct parsing of the first row...")
    
    # Try to work with what we have - use first column for everything if needed
    first_col = df.columns[0]
    
    # If this is a single row with the title,ingredients,etc. combined, try to extract
    if title_col is None and 'title' not in column_mapping:
        df['parsed_title'] = df[first_col].apply(lambda x: x.split('\t')[0] if isinstance(x, str) and '\t' in x else x)
        title_col = 'parsed_title'
        print(f"Created parsed title column: {title_col}")
    
    # Try to extract ingredients from tab-separated data
    if 'ingredients' not in column_mapping:
        try:
            df['parsed_ingredients'] = df[first_col].apply(
                lambda x: ast.literal_eval(x.split('\t')[1]) if isinstance(x, str) and '\t' in x and len(x.split('\t')) > 1 else []
            )
            column_mapping['ingredients'] = 'parsed_ingredients'
            print("Created parsed ingredients column")
        except Exception as e:
            print(f"Could not parse ingredients from combined data: {e}")
    
    # Try to extract directions from tab-separated data
    if 'directions' not in column_mapping:
        try:
            df['parsed_directions'] = df[first_col].apply(
                lambda x: ast.literal_eval(x.split('\t')[2]) if isinstance(x, str) and '\t' in x and len(x.split('\t')) > 2 else []
            )
            column_mapping['directions'] = 'parsed_directions'
            print("Created parsed directions column")
        except Exception as e:
            print(f"Could not parse directions from combined data: {e}")
    
    # Try to extract NER from tab-separated data
    if 'NER' not in column_mapping:
        try:
            df['parsed_NER'] = df[first_col].apply(
                lambda x: ast.literal_eval(x.split('\t')[5]) if isinstance(x, str) and '\t' in x and len(x.split('\t')) > 5 else []
            )
            column_mapping['NER'] = 'parsed_NER'
            print("Created parsed NER column")
        except Exception as e:
            print(f"Could not parse NER from combined data: {e}")

# Feature Engineering for Flan-T5 finetuning
# 1. Add counts for ingredients and directions
for target_col, actual_col in column_mapping.items():
    try:
        df[f'{actual_col.lower()}_count'] = df[actual_col].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0)
        print(f"Added feature '{actual_col.lower()}_count'.")
    except Exception as e:
        print(f"Could not add count feature for {actual_col}: {e}")

# 2. Create text-to-text format for Flan-T5
# For a recipe lookup task:
# - Input: Recipe title or name
# - Output: Ingredients, directions, and source link

# Create input text (prompt) column - just the title
if title_col and title_col in df.columns:
    df['input_text'] = df[title_col].apply(lambda x: f"Generate ingredients and directions for: {x}")
    print(f"Created input text from '{title_col}' column.")
else:
    print("Warning: Could not find title column, using first column as input.")
    df['input_text'] = df.iloc[:, 0].apply(lambda x: 
        f"Generate ingredients and directions for: {x.split('\t')[0] if isinstance(x, str) and '\t' in x else x}")

# Function to safely get ingredients, handling different data formats
def safe_get_ingredients(row, col_mapping):
    ing_col = col_mapping.get('ingredients')
    if ing_col and ing_col in row and isinstance(row[ing_col], (list, tuple)):
        return [f"• {ingredient}" for ingredient in row[ing_col]]
    return ["• No ingredients available"]

# Function to safely get directions
def safe_get_directions(row, col_mapping):
    dir_col = col_mapping.get('directions')
    if dir_col and dir_col in row and isinstance(row[dir_col], (list, tuple)):
        return [f"{i+1}. {step}" for i, step in enumerate(row[dir_col])]
    return ["1. No directions available"]

# Function to safely get NER
def safe_get_ner(row, col_mapping):
    ner_col = col_mapping.get('NER')
    if ner_col and ner_col in row and isinstance(row[ner_col], (list, tuple)):
        return row[ner_col]
    return []

# Create target text column with ingredients, directions, key ingredients (NER) and link
try:
    df['target_text'] = df.apply(
        lambda row: (
            "Ingredients:\n" + 
            "\n".join(safe_get_ingredients(row, column_mapping)) + 
            
            # Add Key Ingredients section from NER data
            "\n\nKey Ingredients:\n" +
            ", ".join(safe_get_ner(row, column_mapping)) +
            
            "\n\nDirections:\n" + 
            "\n".join(safe_get_directions(row, column_mapping)) +
            (f"\n\nSource: {row[link_col]}" if link_col and link_col in row else "") +
            (f"\nWebsite: {row[site_col]}" if site_col and site_col in row else "")
        ),
        axis=1
    )
    print("Created text-to-text format for Flan-T5 finetuning with NER data as Key Ingredients.")
except Exception as e:
    print(f"Error creating target text: {e}")
    print("Attempting simplified target text creation...")
    
    # Fallback to a simpler approach if the above fails
    df['target_text'] = "Could not format recipe data properly. Please check the data format."
    
    # Try to display sample data for debugging
    print("\nSample first row data for debugging:")
    for col in df.columns:
        try:
            print(f"{col}: {df[col].iloc[0]}")
        except:
            print(f"{col}: <error displaying value>")

# Add additional cleaning if needed
# For Flan-T5, it's often good to keep most of the original text structure

print("Preprocessing complete.")

print("Splitting data into train and test sets...")
TEST_SIZE = params.get('preprocess', {}).get('test_size', 0.2)
RANDOM_STATE = params.get('base', {}).get('random_state', 42)

# Split the dataframe
if not df.empty:
    try:
        train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print("Data split successfully.")

        print(f"Train data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")

        print(f"Saving processed data to {PROCESSED_DATA_DIR}...")

        # Ensure the output directory exists
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        # Save full processed data with all columns
        full_processed_path = f'{PROCESSED_DATA_DIR}/full_processed.csv'
        df.to_csv(full_processed_path, index=False)
        print(f"Full processed data saved to {full_processed_path}")
        
        # For T5 finetuning, we only need input_text and target_text columns
        train_df_t5 = train_df[['input_text', 'target_text']]
        test_df_t5 = test_df[['input_text', 'target_text']]
        
        # Save to CSV - only the columns needed for T5 training
        train_df_t5.to_csv(TRAIN_DATA_PATH, index=False)
        test_df_t5.to_csv(TEST_DATA_PATH, index=False)

        print("Processed data saved successfully.")
        
        # Print examples of the formatted data for verification
        print("\nExample of formatted data for Flan-T5:")
        print("Input text example:", train_df_t5['input_text'].iloc[0])
        print("\nTarget text example:", train_df_t5['target_text'].iloc[0])
        
    except ValueError as e:
        print(f"Error during data splitting or saving: {e}")
        print("Please check your data, the test_size parameter, and output directory permissions.")
        exit()
else:
    print("Dataframe is empty after preprocessing. Skipping data splitting and saving.")

