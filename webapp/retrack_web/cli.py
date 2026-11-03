"""Command-line tasks: `flask --app wsgi create-admin` provisions an Admin account (UC-01 constraints)."""
import getpass

import click

from .errors import AppError


def register_cli(app):
    @app.cli.command('create-admin')
    @click.option('--username', prompt=True)
    @click.option('--email', prompt=True)
    def create_admin_command(username, email):
        """Provisions an Admin account. The password is read without echo."""
        from .services.accounts import create_admin
        password = getpass.getpass('Password: ')
        if password != getpass.getpass('Repeat password: '):
            raise click.ClickException('The passwords do not match.')
        try:
            admin = create_admin(username, email, password)
        except AppError as e:
            raise click.ClickException(e.message)
        click.echo(f'Admin {admin["username"]} created ({admin["_id"]}).')
