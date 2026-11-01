import os
import tempfile

# Point the API at a throwaway database before the app module is imported.
os.environ["CHURN_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
