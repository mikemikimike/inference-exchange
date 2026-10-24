# BILL: Billing

### BILL-001: Books balance

MUST: For each charge, the consumer debit equals the provider credit plus the platform fee.

- Principle: P7
- Status: implemented
- Verified by: tests/test_financial_invariants.py::TestBooksBalance::test_per_bill_invariant_across_random_events, tests/test_store.py::TestBilling::test_financial_invariant

### BILL-002: Platform fee is 10 percent

MUST: The platform fee is 10 percent of the charge, rounded down. The provider gets the remainder.

- Principle: P7
- Status: implemented
- Verified by: tests/test_financial_invariants.py::TestPlatformFee::test_platform_fee_is_ten_percent, tests/test_financial_invariants.py::TestProviderSplit::test_provider_gets_ninety_percent

### BILL-003: No negative charges

MUST NOT: A charge has a negative amount.

- Principle: P7
- Status: implemented
- Verified by: tests/test_financial_invariants.py::TestNoNegativeCharges::test_no_negative_amounts

### BILL-004: Minimum charge

MUST: Each billed request costs at least $0.0001.

- Principle: P7
- Status: implemented
- Verified by: tests/test_store.py::TestBilling::test_minimum_charge

### BILL-005: Zero balance stops requests

MUST: If the balance is zero or less, the system rejects the request with HTTP 402.

- Principle: P7
- Status: partial
- Issue: #59

Note: The code rejects at a balance of zero or less. Billing is post-pay, so a balance can go below zero by the cost of one request.

### BILL-006: Served requests are billed

MUST: When a provider completes a request, the system bills the consumer.

- Principle: P7
- Status: implemented
- Verified by: tests/test_mock_provider_e2e.py::test_chat_through_mock_provider_is_served_and_billed

### BILL-007: Partial streams are billed for delivered tokens

MUST: If a stream fails or the consumer cancels it, bill only for the tokens that the provider delivered.

- Principle: P7
- Status: implemented
- Verified by: tests/test_cancellation.py::test_stream_cancellation_bills_delivered_tokens, tests/test_cancellation.py::test_provider_error_bills_delivered_tokens

### BILL-008: Balances persist

MUST: Accounts, balances, and transactions survive a coordinator restart.

- Principle: P7
- Status: implemented
- Verified by: tests/test_store.py::TestPersistence::test_balance_survives_restart, tests/test_store.py::TestPersistence::test_transactions_survive_restart
