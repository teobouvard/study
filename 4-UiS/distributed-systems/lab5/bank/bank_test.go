package bank

import "testing"

const accountNum = 42

var accountTests = []struct {
	txn        Transaction
	wantTxnRes TransactionResult
}{
	{
		Transaction{Op: Balance},
		TransactionResult{
			AccountNum: accountNum,
			Balance:    0,
		},
	},
	{
		Transaction{Op: Deposit, Amount: 1000},
		TransactionResult{
			AccountNum: accountNum,
			Balance:    1000,
		},
	},
	{
		Transaction{Op: Withdrawal, Amount: 501},
		TransactionResult{
			AccountNum: accountNum,
			Balance:    499,
		},
	},
	{
		Transaction{Op: Withdrawal, Amount: 500},
		TransactionResult{
			AccountNum:  accountNum,
			Balance:     499,
			ErrorString: "Not enough funds for withdrawal. Balance: 499 NOK - Requested 500 NOK",
		},
	},
	{
		Transaction{Op: Withdrawal, Amount: 499},
		TransactionResult{
			AccountNum: accountNum,
			Balance:    0,
		},
	},
	{
		Transaction{Op: Deposit, Amount: -100},
		TransactionResult{
			AccountNum:  accountNum,
			Balance:     0,
			ErrorString: "Can't deposit negative amount (-100 NOK)",
		},
	},
	{
		Transaction{Op: 4, Amount: 1234},
		TransactionResult{
			AccountNum:  accountNum,
			Balance:     0,
			ErrorString: "Unknown transaction type (4)",
		},
	},
	{
		Transaction{Op: Deposit, Amount: 1500},
		TransactionResult{
			AccountNum: accountNum,
			Balance:    1500,
		},
	},
}

func TestAccount(t *testing.T) {
	a := Account{Number: accountNum, Balance: 0}
	for i, at := range accountTests {
		gotTxnRes := a.Process(at.txn)
		if gotTxnRes != at.wantTxnRes {
			t.Errorf("Test %d:\nwant:\t%v\ngot:\t%v", i, at.wantTxnRes, gotTxnRes)
		}
	}
}
