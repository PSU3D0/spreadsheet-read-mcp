#!/bin/sh
set -eu

BIN="/usr/local/bin/agent-spreadsheet-mcp"

if [ "$(id -u)" -ne 0 ]; then
  exec "$BIN" "$@"
fi

workspace_root="${SPREADSHEET_MCP_WORKSPACE:-/data}"

# Best-effort parse of --workspace-root from args.
prev=""
for a in "$@"; do
  if [ "$prev" = "--workspace-root" ]; then
    workspace_root="$a"
    prev=""
    continue
  fi
  prev="$a"
done

uid="${SPREADSHEET_MCP_RUN_UID:-}"
gid="${SPREADSHEET_MCP_RUN_GID:-}"

if [ -z "$uid" ] || [ -z "$gid" ]; then
  auto="${SPREADSHEET_MCP_AUTO_UID_GID:-true}"
  if [ "$auto" = "true" ] && [ -d "$workspace_root" ]; then
    owner="$(stat -c '%u:%g' "$workspace_root" 2>/dev/null || true)"
    if [ -n "$owner" ]; then
      uid="${owner%%:*}"
      gid="${owner##*:}"
    fi
  fi
fi

uid="${uid:-1000}"
gid="${gid:-1000}"

# Create a per-UID LibreOffice user installation.
# Prefer cloning the pre-initialized template profile (created during image build),
# which includes the full set of defaults plus our macro security settings.
lo_profile_root="/tmp/agent-spreadsheet-mcp-lo-profile-${uid}"
template_root="/root/.config/libreoffice/4"
if [ -d "${template_root}/user" ] && [ ! -f "${lo_profile_root}/user/registrymodifications.xcu" ]; then
  rm -rf "${lo_profile_root}" 2>/dev/null || true
  cp -a "${template_root}" "${lo_profile_root}" 2>/dev/null || cp -r "${template_root}" "${lo_profile_root}"
fi

# Fallback: copy just our macro files into an empty profile.
lo_user_dir="${lo_profile_root}/user"
if [ ! -f "${lo_user_dir}/basic/Standard/Module1.xba" ]; then
  mkdir -p "${lo_user_dir}/basic/Standard" "${lo_user_dir}/basic"
  if [ -f "/etc/libreoffice/4/user/basic/Standard/Module1.xba" ]; then
    cp "/etc/libreoffice/4/user/basic/Standard/Module1.xba" "${lo_user_dir}/basic/Standard/Module1.xba"
  fi
  if [ -f "/etc/libreoffice/4/user/basic/Standard/script.xlb" ]; then
    cp "/etc/libreoffice/4/user/basic/Standard/script.xlb" "${lo_user_dir}/basic/Standard/script.xlb"
  fi
  if [ -f "/etc/libreoffice/4/user/basic/script.xlc" ]; then
    cp "/etc/libreoffice/4/user/basic/script.xlc" "${lo_user_dir}/basic/script.xlc"
  fi
  if [ -f "/etc/libreoffice/4/user/registrymodifications.xcu" ]; then
    cp "/etc/libreoffice/4/user/registrymodifications.xcu" "${lo_user_dir}/registrymodifications.xcu"
  fi
fi

chown -R "$uid:$gid" "$lo_profile_root" 2>/dev/null || true
export SPREADSHEET_MCP_LIBREOFFICE_USER_INSTALLATION="$lo_profile_root"

exec gosu "$uid:$gid" "$BIN" "$@"
