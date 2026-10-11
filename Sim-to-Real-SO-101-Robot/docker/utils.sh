source /root/env 2>/dev/null

setenv() {
  local key val
  if [ -z "$1" ]; then
    echo "Usage:"
    echo "  envset KEY VALUE"
    echo "  envset KEY=VALUE"
    echo "  envset -d KEY"
    echo "  envset            — list current env file"
    cat /root/env 2>/dev/null
    return 1
  fi
  if [ "$1" = "-d" ]; then
    sed "/^export ${2}=/d" /root/env > /tmp/env.tmp && cat /tmp/env.tmp > /root/env
    unset "$2"
    echo "Removed $2"
    return 0
  fi
  if [ -n "$2" ]; then
    key="$1"
    val="$2"
  else
    key="${1%%=*}"
    val="${1#*=}"
  fi
  if grep -q "^export ${key}=" /root/env 2>/dev/null; then
    sed "s|^export ${key}=.*|export ${key}=${val}|" /root/env > /tmp/env.tmp && cat /tmp/env.tmp > /root/env
  else
    printf '\n%s\n' "export ${key}=${val}" >> /root/env
  fi
  source /root/env
  echo "${key}=${val}"
}
