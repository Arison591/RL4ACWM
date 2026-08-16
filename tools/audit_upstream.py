#!/usr/bin/env python3
import json

from tempflow_video.runtime.integrity import audit_upstream

if __name__ == "__main__":
    print(json.dumps(audit_upstream(), indent=2))

