# mod_dc_logger.py
import BigWorld

def _log(msg):
    try:
        # python.log 里可见
        print('[dc.logger] ' + msg)
    except:
        pass

_log('loaded!')
