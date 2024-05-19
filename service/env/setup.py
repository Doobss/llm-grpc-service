from logging import getLogger, basicConfig
# from gpsecore.environment import initialize_environment
logger = getLogger()
logger.propagate = False


def queue_env():
    logger.info('Setting queue up environment')
    # initialize_environment(scopes=['globals', 'databases', 'queues', 'routes', 'services/llm'])


def socket_env():
    logger.info('Setting socket_server up environment')
    # initialize_environment(scopes=['globals', 'databases', 'queues', 'routes', 'services/llm'])