Data Release Process
====================

SDC Release Process
-------------------

The SDC will require a GitHub ticket containing the following information:

- Which instrument
- What date range will be made public
- Withhold files attached to the ticket, if any

  .. note:: See :ref:`withhold-file` for more details on what that file contains.

- Release notes attached to the ticket, if any
- A release due date on the ticket

The SDC will review, plan, and process release tickets, then publish data through the release API. The API
backend will switch the release flag to ``true``, indicating that the file is made public.
Once that flag is updated, any user will be able to view data on the IMAP SDC website
without login credentials.

Metadata Validation
-------------------

Metadata fields must comply with both ISTP and SPDF requirements.

The project will produce an easy-to-read document containing the metadata for all IMAP
products, making it easier for Instrument Teams to verify that metadata are accurate and
contain the desired level of information about their products.

.. _withhold-file:

Withhold File
-------------

The SDC supports withholding data from public release. Follow the filename convention and
instructions below.

Filename Convention
~~~~~~~~~~~~~~~~~~~

.. code-block:: none

    imap_<instrument>_<description>_data-release-<###>_<version>.<extension>

Similar to the ancillary filename convention defined in the IMAP SDC Documentation, release
files follow the same format with the following restrictions:

- The ``<description>`` field only accepts the value ``withhold``.
- The ``<extension>`` must be ``.txt``.

File Contents
~~~~~~~~~~~~~

A list of filenames — covering both ancillary and science data — that will not be made
public.

SPDF Archival and Release Process
----------------------------------

SPDF has confirmed that it will archive and make data available to the public through
its user interface by downloading public IMAP data using ``imap-data-access`` features.

Reprocessing Before Public Release
------------------------------------

Types of Reprocessing
~~~~~~~~~~~~~~~~~~~~~

Events that would trigger reprocessing include:

- **Code updates** — requires manual reprocessing.
- **Ancillary file upload** — automatically triggers reprocessing for date ranges covered by
  the file.
- **SPICE file updates** — may trigger automatic reprocessing or may require manual
  reprocessing, depending on dependency definitions.

Reprocessing Requests
~~~~~~~~~~~~~~~~~~~~~

The SDC will require a GitHub ticket containing the following information:

- Which instrument
- What product or data level
- What date range to reprocess
- A reprocessing due date on the ticket

The SDC will review reprocessing tickets in release planning sessions or weekly planning meetings.
