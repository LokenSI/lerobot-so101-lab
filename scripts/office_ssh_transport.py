"""Portable explicit SSH configuration; no instance discovery or saved account data."""
import os
from pathlib import Path

def connection():
    return os.environ['OFFICE_SSH_TARGET'],int(os.environ.get('OFFICE_SSH_PORT','22'))

def options():
    key=Path(os.environ['OFFICE_SSH_KEY']);known=Path(os.environ['OFFICE_SSH_KNOWN_HOSTS'])
    if not key.is_file() or not known.is_file():raise FileNotFoundError('Supply an existing SSH key and verified known-hosts file')
    return ['-i',str(key),'-o','UserKnownHostsFile='+str(known),'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes']
