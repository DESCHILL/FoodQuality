import unittest
from scripts.import_foodclass_expansion import select_groups


class FoodclassExpansionTests(unittest.TestCase):
    def row(self,i,condition='fresh'):
        return dict(sha256=f'{i:064x}',food='apple',condition=condition)

    def test_transitive_overlap_excludes_whole_new_group(self):
        rows=[self.row(i) for i in range(3)]
        selected,audit=select_groups(rows,[0,15,255],1)
        self.assertEqual(selected,[])
        self.assertEqual(audit['overlap_with_previous_sources'],2)

    def test_conflicting_labels_excluded(self):
        selected,audit=select_groups([self.row(1),self.row(2,'spoiled')],[0,1],0)
        self.assertEqual(selected,[])
        self.assertEqual(audit['conflicting_source_labels'],2)

    def test_one_representative_and_deterministic_split(self):
        rows=[self.row(4),self.row(3)]
        selected,audit=select_groups(rows,[0,1],0)
        reverse,_=select_groups(rows[::-1],[1,0],0)
        self.assertEqual(selected,reverse)
        self.assertEqual(len(selected),1)
        self.assertEqual(audit['redundant_new_images'],1)


if __name__=='__main__':unittest.main()
