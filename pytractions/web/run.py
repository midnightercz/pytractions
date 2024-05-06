import os
import streamlit
import streamlit.web.bootstrap
from streamlit import config as _config

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname)

_config.set_option("server.headless", True)
args = []

#streamlit.cli.main_run(filename, args)
streamlit.web.bootstrap.load_app({})

streamlit.web.bootstrap.run(filename,'',args, flag_options={})
