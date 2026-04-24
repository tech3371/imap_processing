Weekly Planning Session
=======================

The SDC holds a weekly planning session to review progress, surface blockers, and
coordinate work across the team.

`Current Month Milestone GitHub Board <https://github.com/orgs/IMAP-Science-Operations-Center/projects/2/views/54>`_

Agenda
------

Ticket Status Updates
~~~~~~~~~~~~~~~~~~~~~

Dedicate 10 minutes for everyone to go through the following steps on the
`Current Month Milestone Board`:

1. Identify any tickets you are blocked on and update their status to **Blocked** if not
   already set. Add a comment describing what or who you are waiting on to move forward.
2. Update status from **In Progress** to **Open PR** if you have an open PR for a ticket
   that is ready for review.
3. Update status from **TODO** to **In Progress** if you are actively working on a ticket.
4. Create a new ticket with the correct status for any unplanned work (e.g., bug fixes)
   from last week, or bring ticket into the current month milestone if ticket exists already.
5. Close any open tickets whose work is complete or cancelled.

Changes Review
~~~~~~~~~~~~~~

- Were any tasks added to or removed from the current milestone? Create a ticket if one
  does not exist already.

  - Does the ticket have a priority and story points assigned?

- Did the scope of work change on any current milestone task (e.g., a minor task whose
  estimate was greater than planned)?

- For new requests, add the **Change Requests** and **Phase E: Parent Issue** labels to the ticket for review. See the
  `Process for Change Requests` section below for more details on communication and review of change requests. Once
  reviewed by the project, the ticket will be planned into current or a future milestone based on priority and resource availability.

**Process for Change Requests**

1. The instrument team meets internally to finalize all details needed for the change request.
    A new algorithm document with specifics is the ideal outcome, but at minimum, the team
    should have fully developed and validated their findings before bringing any proposed
    algorithm to the project.

2. The change request ticket and algorithm document are then sent in a single email to the following stakeholders:

   - **Project leadership** (SOC Lead, IMAP SDS Lead, and/or ENA Lead) — members of IMAP leadership with authority to approve, reject, or prioritize the change.
   - **Instrument Team lead** (Instrument Lead or Algorithm experts) — the point of contact who owns scientific knowledge of the change.
   - **SDC instrument lead** (Instrument experts at SDC) — the point of contact who knows implementation details and can provide input on work effort and estimation.
   - **SDC Lead** (Work Planning management) — responsible for resource and work planning at the SDC.

Blockers
~~~~~~~~

- Identify any blockers and create a new ticket if one does not exist already:

  - Anything blocked by a PR review
  - Anything blocked by features on SDC infrastructure
  - Anything blocked by outside entities. Eg. instrument teams

- If blocked, update the ticket status to **Blocked** for review.

Completed Work Summary
~~~~~~~~~~~~~~~~~~~~~~

Each team member gives a brief summary of completed work to help the team understand
what the SDC has accomplished as a whole. This can be done by walking through current
work in progress — for example, a quick status update on open or recently merged PRs.
These walkthroughs should be short lightning talks; save follow-ups for Slack or
individual meetings.
