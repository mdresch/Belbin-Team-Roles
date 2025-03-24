# prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_file="sentences.csv", train_file="train.csv", validation_file = "validation.csv", test_file="test.csv"):
    """Loads, cleans, and splits data into training, validation, and test sets."""

    try:
        # --- Debugging: Read and print raw lines ---
        print("DEBUG: Reading raw lines from the file:")
        with open(input_file, 'r', encoding='utf-8') as f:  # Use utf-8 encoding
            for i, line in enumerate(f):
                print(f"  Line {i+1}: {line.rstrip()}")  # Print each line (remove trailing newline)
        print("DEBUG: Raw lines read complete.\n")
        # --- End Debugging ---

        # Load data with explicit header names, and comma as separator
        df = pd.read_csv(
            input_file,
            sep=',',           # Explicitly set the separator to a comma.
            header=None,       # No header row in the input file.
            names=['sentence', 'belbin_role', 'label'],  # Provide column names.  <- CHANGED!
            encoding='utf-8',   # Specify UTF-8 encoding (VERY IMPORTANT).
            quotechar='"',     # Handle quotes correctly.
            skipinitialspace=True, # Added to skip spaces after comma
        )

        # Strip leading/trailing whitespace from *both* columns.
        df['sentence'] = df['sentence'].str.strip()
        df['label'] = df['label'].str.strip()
        df['belbin_role'] = df['belbin_role'].str.strip() # also strip spaces


        print("DEBUG: DataFrame loaded:\n", df.head())  # Print first few rows
        print("DEBUG: DataFrame info:")
        df.info()  # Print DataFrame info (dtypes, non-null counts)
        print()

        # Check for missing values *before* any other operations
        if df.isnull().values.any():
            print("DEBUG: Missing values found *before* label processing:")
            print(df.isnull().sum())  # Show count of missing values per column
            print(df[df.isnull().any(axis=1)])  # Show rows with *any* missing values
            raise ValueError("Missing values found in the dataset. Please clean your data.")

        # Check for valid labels and convert to lowercase
        valid_labels = ["positive", "neutral", "negative"]
        df['label'] = df['label'].str.lower()  # Convert labels to lowercase
        invalid_labels = df[~df['label'].isin(valid_labels)]
        if not invalid_labels.empty:
            print("Invalid labels found:\n", invalid_labels)
            raise ValueError("Invalid labels found in the dataset. Labels must be 'positive', 'neutral', or 'negative'.")

        # Check for valid belbin roles. KEEP SPACES.
        valid_roles = ['Shaper', 'Implementer', 'Completer Finisher', 'Co-ordinator', 'Teamworker', 'Resource Investigator', 'Plant', 'Monitor Evaluator', 'Specialist']
        invalid_roles = df[~df['belbin_role'].isin(valid_roles)]
        if not invalid_roles.empty:
             print("Invalid roles found:\n", invalid_roles)
             raise ValueError("Invalid Belbin roles found. Check for correct spelling and spacing.")


        # Remove duplicate sentences (keep the first occurrence)
        df = df.drop_duplicates(subset=['sentence'], keep='first')
        print(f"DEBUG: Number of rows after removing duplicates: {len(df)}")


        # Split the data: 90% train, 10% test.  NO STRATIFICATION.
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
        # IMPORTANT:  We are no longer using stratify.  This is a last resort.
        # No validation set.

        # Save the splits to separate CSV files
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        #Removed validation file
        print(f"Data preparation complete.  Created: {train_file}, {test_file}") # Removed validation.csv
        print(f"Train set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")


    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please make sure it exists.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: '{input_file}' is empty. Please provide a valid CSV file.")
        exit(1)
    except pd.errors.ParserError:
        print(f"Error: '{input_file}' could not be parsed.  Is it a valid CSV file?")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)



if __name__ == "__main__":
    prepare_data()