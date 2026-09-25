import ipaddress
from urllib.parse import urlsplit


class BaseUrlConfigError(ValueError):
    pass


def is_private_network_host(hostname: str) -> bool:
    """Check whether a host may use plain HTTP for local access."""
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_private
    except ValueError:
        pass
    # Unqualified single-label hostnames (e.g. Docker Compose service names
    # like "ollama") cannot resolve as public internet FQDNs.
    return "." not in hostname


def validate_base_url(base_url: str, env_var_name: str) -> str:
    """Require HTTPS unless the host is local or private."""
    parsed = urlsplit(base_url)
    if parsed.scheme == "https":
        return base_url
    if parsed.scheme == "http" and parsed.hostname is not None and is_private_network_host(parsed.hostname):
        return base_url
    raise BaseUrlConfigError(
        f"{env_var_name} must use HTTPS for remote/public hosts (got: {base_url!r}); "
        "plain HTTP is only allowed for loopback, private-network, or unqualified internal hostnames."
    )
