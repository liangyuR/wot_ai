# mod_dc_position.py
import BigWorld
import Math
import traceback

LOG_PREFIX = '[dc.position]'
REPORT_INTERVAL = 1.0
MIN_MOVE_DISTANCE = 0.5  # meters

_callback_id = None
_last_pos = None
_notice_scheduled = False


def _log(msg):
    try:
        print(LOG_PREFIX + ' ' + msg)
    except Exception:
        pass


def _get_player_vehicle():
    player = BigWorld.player()
    if player is None:
        return None
    if not hasattr(player, 'playerVehicleID'):
        return None
    if player.playerVehicleID is None:
        return None
    try:
        return BigWorld.entity(player.playerVehicleID)
    except Exception:
        return None


def _format_vec(v):
    return '%.2f,%.2f,%.2f' % (v.x, v.y, v.z)


def _should_log(pos):
    global _last_pos
    if _last_pos is None:
        _last_pos = Math.Vector3(pos)
        return True
    dx = pos.x - _last_pos.x
    dy = pos.y - _last_pos.y
    dz = pos.z - _last_pos.z
    if (dx * dx + dy * dy + dz * dz) >= (MIN_MOVE_DISTANCE * MIN_MOVE_DISTANCE):
        _last_pos = Math.Vector3(pos)
        return True
    return False


def _tick():
    global _callback_id
    try:
        vehicle = _get_player_vehicle()
        if vehicle is not None:
            pos = vehicle.position
            if pos is not None and _should_log(pos):
                _log('pos=' + _format_vec(pos))
    except Exception:
        _log('error: ' + traceback.format_exc().splitlines()[-1])
    finally:
        _callback_id = BigWorld.callback(REPORT_INTERVAL, _tick)


def _try_show_message():
    global _notice_scheduled
    try:
        import messenger
        gui = messenger.MessengerEntry.g_instance.gui
        if gui is not None:
            gui.addClientMessage('dc.position loaded', True)
            _notice_scheduled = False
            return
    except Exception:
        pass

    _notice_scheduled = True
    BigWorld.callback(0.5, _try_show_message)


def start():
    global _callback_id
    if _callback_id is None:
        _callback_id = BigWorld.callback(REPORT_INTERVAL, _tick)
    if not _notice_scheduled:
        _try_show_message()


_log('loaded')
start()
