# --- 1. Imports ---
# You will need pandas for this module
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

class CSVManager:
    # --- 2. The Constructor (__init__) ---
    def __init__(self, csv_path: str):
        """
        Initializes the CSVManager.
        - Stores the path to the CSV file.
        - Loads the CSV into a pandas DataFrame.
        """ 
        self.csv_path = csv_path
        try:
            # Load the CSV file into a DataFrame.
            # A DataFrame is like a powerful spreadsheet in your code.
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"Error: The file at {self.csv_path} was not found.")
            # We can create a default empty DataFrame to prevent crashes.
            self.df = pd.DataFrame() # Create an empty DataFrame with the expected columns

    # --- 3. Get the Next Lead to Call ---
    def get_next_lead(self):
        """
        Finds the first lead with the status 'New' and returns it.
        Returns None if no new leads are found.
        """
        # A lead is a row in our DataFrame.
        # We need to filter the DataFrame to find rows where 'lead_status' is 'New'.
        other_leads = self.df[(self.df['lead_status'].str.lower() == 'other') & (self.df['attempt_count'] < 3)]

        if not other_leads.empty:
            # Get the first lead from the filtered list
            # .iloc[0] selects the first row
            lead = other_leads.iloc[0]
            return lead.to_dict() # Return the lead's data as a dictionary
        else:
            return None

    # --- 4. Update a Lead's Status ---
    # in CSVManager class

    def update_lead(self, lead_id, new_status: str, summary: str = "", conversation_history: str = ""): # Add new parameter
        """
        Finds a lead by their 'lead_id' and updates their information.
        """
        lead_index = self.df.index[self.df['lead_id'] == lead_id].tolist()

        if lead_index:
            idx = lead_index[0]
            self.df.loc[idx, 'lead_status'] = new_status
            self.df.loc[idx, 'call_summary'] = summary
            self.df.loc[idx, 'attempt_count'] += 1
            self.df.loc[idx, 'last_update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            

            self.save_to_csv()
            print(f"Updated lead {lead_id} with status: {new_status}")
        else:
            print(f"Error: Lead with ID {lead_id} not found.")

    # --- 5. Save Changes to the CSV File ---
    def save_to_csv(self):
        """
        Saves the current state of the DataFrame back to the CSV file.
        """
        # The to_csv() method does this for us.
        # 'index=False' prevents pandas from writing the DataFrame index as a column.
        self.df.to_csv(self.csv_path, index=False)


# --- 6. Test Block ---
if __name__ == "__main__":
    # Create a dummy CSV for testing if it doesn't exist
    
    # Path to your leads file
    leads_file_path = "/home/pranavnamburi/Desktop/pipecat-demo/lead_agent/data/leads.csv"
    
    csv_manager = CSVManager(leads_file_path)

    if not csv_manager.df.empty:
        # Get a lead
        next_lead = csv_manager.get_next_lead()
        if next_lead:
            print("Next lead to call:")
            print(next_lead)

            # Simulate a call and update the lead
            lead_id_to_update = next_lead['lead_id']
            csv_manager.update_lead(
                lead_id=lead_id_to_update,
                new_status="Contacted",
                summary="User asked about pricing. Seemed interested."
            )
        else:
            print("No new leads to call.")