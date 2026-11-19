#!/bin/bash
# Push every local branch to origin as roy/archive/<branch> only (never to main). Needs GHT env.
set -u; cd /home/smc/workspace/roy/unieinfra
G=(git -c credential.helper= -c 'credential.helper=!f(){ echo username=x-access-token; echo "password=$GHT"; }; f')
specs=(); for b in $(git for-each-ref --format='%(refname:short)' refs/heads); do specs+=("+refs/heads/$b:refs/heads/roy/archive/$b"); done
"${G[@]}" push origin "${specs[@]}" 2>&1 | tail -60
