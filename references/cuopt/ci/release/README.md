# For cuOpt update provide only next tag withiout 'v' and 'a' elements in tag

ci/release/update-version-cuopt.sh 24.03.00

# For rapids update provide both current and previous tags withiout 'v' and 'a' elements in tag

ci/release/update-version-rapids.sh 23.12.00 24.03.00
