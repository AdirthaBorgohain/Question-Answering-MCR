mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableWebsocketCompression = false\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
