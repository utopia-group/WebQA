import csv
import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials
from lib.config import DEFAULT_GOOGLE_CREDENTIALS


class GoogleService:
    """
    Utility class to invoke google services for spreadsheets. Used to
    access various spreadsheets for the project.
    """
    def __init__(self, credential_file=DEFAULT_GOOGLE_CREDENTIALS):
        """
        Authorize a new google service account using given credentials.
        """
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            credential_file, scope)
        self.gc = gspread.authorize(credentials)

    def open_file(self, link):
        self.sh = self.gc.open_by_url(link)
        return self.sh

    def select_wk(self, title):
        self.wk = self.sh.worksheet(title)
        return self.wk

    def insert_cell(self, column, row, info):
        """
        Used to insert into a cell given a named row / column rather
        than indices.
        """
        col_cell = self.wk.find(column)
        row_cell = self.wk.find(row)
        self.wk.update_cell(row_cell.row, col_cell.col, info)
