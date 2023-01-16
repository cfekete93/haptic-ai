import json


def get_coins(url: str = ''):
    """Return a list of coins"""
    url = url.strip()
    if url == '':
        url = './coins.json'
        with open(url, 'r') as file:
            coins = json.loads(file.read())
    else:
        raise NotImplementedError('get_coins is not implemented to make API calls yet')
    return coins


def test():
    coins = get_coins()


if __name__ != '__main__':
    coins = get_coins()


if __name__ == '__main__':
    test()
