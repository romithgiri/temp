
import rsa

class Client:

    def __init__(self):
        self._private_key = None
        self._public_key = None

    def generate_keys(self, key_size=2048):
        """Generates a pair of RSA keys."""
        self._private_key, self._public_key = rsa.newkeys(key_size)

    def get_private_key(self):
        """Returns the private key."""
        return self._private_key

    def get_public_key(self):
        """Returns the public key."""
        return self._public_key

client = Client()
client.generate_keys()

print("Public key:\n", client.get_public_key())
print("\n\n\n\n\nPrivate key:\n", client.get_private_key())
