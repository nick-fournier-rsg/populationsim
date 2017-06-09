# PopulationSim
# See full license in LICENSE.txt.

import logging
import orca

logger = logging.getLogger(__name__)


def setting(key, default=None):

    settings = orca.get_injectable('settings')

    return settings.get(key, default)


def data_dir_from_settings():
    """
    legacy strategy foir specifying data_dir is with orca injectable.
    Calling this function provides an alternative by reading it from settings file
    """

    # FIXME - not sure this plays well with orca
    # it may depend on when file with orca decorator is imported

    data_dir = setting('data_dir', None)

    if data_dir:
        orca.add_injectable('data_dir', data_dir)
    else:
        data_dir = orca.get_injectable('data_dir')

    logger.info("data_dir: %s" % data_dir)
    return data_dir
